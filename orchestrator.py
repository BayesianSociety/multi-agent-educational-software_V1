#!/usr/bin/env python3
"""Deterministic Codex orchestrator for multi-step software delivery.

Design A is implemented as the default mode.
Design B features are enabled with --design-b.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------- constants -----------------------------------

ERROR_CODEX_FLAG_UNSUPPORTED = "CODEX_CLI_FLAG_UNSUPPORTED"
ERROR_TRANSIENT_TRANSPORT = "CODEX_TRANSIENT_TRANSPORT"
ERROR_INVARIANT_ORCH_CHANGED = "INVARIANT_ORCHESTRATOR_MUTATED"
ERROR_INVARIANT_GIT_CHANGED = "INVARIANT_GIT_MUTATED"
ERROR_ALLOWLIST = "ALLOWLIST_VIOLATION"
ERROR_FORBIDDEN_PATH = "FORBIDDEN_PATH_TOUCHED"
ERROR_CAP_FILES = "CHANGE_CAP_FILES"
ERROR_CAP_BYTES = "CHANGE_CAP_BYTES"
ERROR_CAP_DELETES = "CHANGE_CAP_DELETES"
ERROR_TEST_CMD_MISSING = "TEST_CMD_MISSING"
ERROR_BRIEF_YAML_INVALID = "BRIEF_YAML_INVALID"
ERROR_BACKEND_UNUSED = "BACKEND_UNUSED"
ERROR_PERSISTENCE_MISSING = "PERSISTENCE_MISSING"
ERROR_E2E_MISSING = "E2E_MISSING"
ERROR_PLACEHOLDER_IMPL = "PLACEHOLDER_IMPL"
ERROR_UX_BASELINE_FAIL = "UX_BASELINE_FAIL"
ERROR_PROMPT_QUALITY_INSUFFICIENT = "PROMPT_QUALITY_INSUFFICIENT"

TRANSIENT_PATTERNS = (
    "stream disconnected",
    "error sending request",
    "channel closed",
)

REQUIRED_FILES = [
    "REQUIREMENTS.md",
    "TEST.md",
    "AGENT_TASKS.md",
    "README.md",
    "RUNBOOK.md",
]

REQUIRED_DIRS = ["design", "frontend", "backend", "tests"]

DESIGN_B_REQUIRED_DIRS = ["prompts", ".codex/skills"]

FORBIDDEN_SUBSTRINGS = [
    "ignore validators",
    "bypass allowlists",
    "write outside allowed paths",
    "mark step as done even if tests fail",
    "modify .orchestrator",
]

DEFAULT_LIMITS = {
    "max_changed_files": 60,
    "max_total_bytes_changed": 500_000,
    "max_deleted_files": 0,
}


# ----------------------------- data models ---------------------------------


@dataclass
class StepConfig:
    name: str
    role: str
    allowlist: List[str]
    max_attempts: int = 3
    can_delete: bool = False
    run_codex: bool = True


@dataclass
class AttemptResult:
    step_name: str
    attempt: int
    variant_id: str
    exit_code: int
    changed_paths: List[str]
    validation_codes: List[str]
    rollback: bool
    retries_used: int
    fixer_runs: int
    skill_used: bool
    skill_path: str
    skill_hash: str
    skill_excerpt_mode: str
    prompt_epoch_id: str


# ----------------------------- orchestrator --------------------------------


class Orchestrator:
    def __init__(self, repo_root: Path, design_b: bool, mode: str) -> None:
        self.repo_root = repo_root
        self.design_b = design_b
        self.mode = mode

        self.orch_root = self.repo_root / ".orchestrator"
        self.runs_root = self.orch_root / "runs"
        self.evals_root = self.orch_root / "evals"
        self.policy_path = self.orch_root / "policy.json"
        self.prompt_template_root = self.orch_root / "prompt_templates"

        self.project_brief_path = self.repo_root / "PROJECT_BRIEF.md"
        self.project_brief_yaml_path = self.repo_root / "PROJECT_BRIEF.yaml"

        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_root = self.runs_root / self.run_id

        self.flag_support = self.detect_codex_flags()
        self.policy = self.load_policy()
        self.project_config = self.load_project_config()

        self.backend_required = True
        if self.project_brief_yaml_path.exists():
            self.backend_required = self.project_config.get("backend_required", True)

        self.locked_files = {
            "PROJECT_BRIEF.md": False,
            "PROJECT_BRIEF.yaml": False,
            "AGENTS.md": False,
        }

        self.prompt_map: Dict[str, Any] = {"run_id": self.run_id, "steps": []}

    # ----------------------------- core lifecycle --------------------------

    def run(self) -> int:
        self.ensure_orchestrator_dirs()
        self.log(f"run_id={self.run_id}")
        self.log(f"design_b={self.design_b} mode={self.mode}")

        steps = self.build_pipeline_steps()

        for step in steps:
            ok = self.run_step_with_retries(step)
            if not ok:
                self.log(f"step_failed={step.name}")
                if self.design_b:
                    self.write_eval(score=-1, details={"failed_step": step.name})
                self.flush_prompt_map()
                return 1
            if step.name == "release_engineer":
                if self.project_brief_path.exists():
                    self.locked_files["PROJECT_BRIEF.md"] = True
                if self.project_brief_yaml_path.exists():
                    self.locked_files["PROJECT_BRIEF.yaml"] = True
                if self.design_b and (self.repo_root / "AGENTS.md").exists():
                    self.locked_files["AGENTS.md"] = True

        product_codes = self.run_product_acceptance_gate()
        if product_codes:
            self.log(f"product_acceptance_failed={product_codes}")
            if self.design_b:
                self.write_eval(score=-1, details={"product_acceptance": product_codes})
            self.flush_prompt_map()
            return 1

        score = self.compute_score(total_retries=0, fixer_runs=0)
        if self.design_b:
            self.write_eval(score=score, details={"status": "pass"})

        self.flush_prompt_map()
        self.log("pipeline_complete=1")
        return 0

    # ----------------------------- setup ----------------------------------

    def detect_codex_flags(self) -> Dict[str, bool]:
        result = self.run_cmd(["codex", "exec", "--help"], check=False)
        text = (result["stdout"] + "\n" + result["stderr"]).lower()
        return {
            "experimental_json": "--experimental-json" in text,
            "output_schema": "--output-schema" in text,
            "ask_for_approval": "--ask-for-approval" in text,
            "config": "--config" in text,
        }

    def ensure_orchestrator_dirs(self) -> None:
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.evals_root.mkdir(parents=True, exist_ok=True)
        self.run_root.mkdir(parents=True, exist_ok=True)
        (self.run_root / "steps").mkdir(parents=True, exist_ok=True)

    def load_policy(self) -> Dict[str, Any]:
        if self.policy_path.exists():
            return json.loads(self.policy_path.read_text(encoding="utf-8"))

        policy = {
            "selection_strategy": "ucb1",
            "bootstrap_min_trials_per_variant": 3,
            "ucb_c": 1.0,
            "commit_window_runs": 10,
            "elim_min_trials": 6,
            "elim_min_mean_clean": 0.1,
            "elim_max_failure_rate": 0.9,
            "roles": {},
            "constraint_patches": {},
            "step_limits": {},
            "unavailable_codex_flags": [],
        }
        self.write_json(self.policy_path, policy)
        return policy

    def load_project_config(self) -> Dict[str, Any]:
        if not self.project_brief_yaml_path.exists():
            return {}
        try:
            data = json.loads(self.project_brief_yaml_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            self.fail_fast(ERROR_BRIEF_YAML_INVALID, f"PROJECT_BRIEF.yaml parse failed: {exc}")

        if not isinstance(data, dict) or not isinstance(data.get("project_type"), str):
            self.fail_fast(ERROR_BRIEF_YAML_INVALID, "PROJECT_BRIEF.yaml missing project_type string")
        return data

    def build_pipeline_steps(self) -> List[StepConfig]:
        steps = [
            StepConfig(
                name="release_engineer",
                role="Release Engineer",
                allowlist=[
                    "REQUIREMENTS.md",
                    "TEST.md",
                    "AGENT_TASKS.md",
                    "README.md",
                    "RUNBOOK.md",
                    "docker-compose.yml",
                    ".env.example",
                    ".gitignore",
                    "PROJECT_BRIEF.md",
                    "PROJECT_BRIEF.yaml",
                ]
                + (["AGENTS.md"] if self.design_b else []),
            ),
            StepConfig(
                name="planner",
                role="Planner",
                allowlist=[".pipeline_plan.json", ".pipeline_plan_schema.json"],
            ),
            StepConfig(
                name="requirements",
                role="Requirements Analyst",
                allowlist=["REQUIREMENTS.md", "AGENT_TASKS.md"],
            ),
            StepConfig(
                name="designer",
                role="UX / Designer",
                allowlist=["design/**", "REQUIREMENTS.md"],
            ),
            StepConfig(
                name="frontend",
                role="Frontend Dev",
                allowlist=["frontend/**", "tests/**"],
            ),
        ]
        if self.backend_required:
            steps.append(
                StepConfig(
                    name="backend",
                    role="Backend Dev",
                    allowlist=["backend/**", "tests/**", ".env.example", "docker-compose.yml"],
                )
            )
        steps.extend(
            [
                StepConfig(
                    name="qa",
                    role="QA Tester",
                    allowlist=["tests/**", "TEST.md"],
                ),
                StepConfig(
                    name="docs",
                    role="Docs Writer",
                    allowlist=["README.md", "RUNBOOK.md"],
                ),
            ]
        )

        if self.design_b:
            if self.prompt_library_missing_or_empty():
                steps.insert(
                    0,
                    StepConfig(
                        name="prompt_library_bootstrap",
                        role="Prompt Library Bootstrap",
                        allowlist=["prompts/**", ".codex/skills/**"],
                    ),
                )

        return steps

    # ----------------------------- step execution --------------------------

    def run_step_with_retries(self, step: StepConfig) -> bool:
        max_attempts = step.max_attempts
        for attempt in range(1, max_attempts + 1):
            self.log(f"step={step.name} attempt={attempt}/{max_attempts}")

            variant_id, variant_text, prompt_epoch_id = self.select_variant(step)
            skill = self.resolve_skill(step)
            prompt = self.build_step_prompt(step, variant_text, skill)

            pre = self.capture_pre_window_snapshot()
            code, out, err, retries_used = self.invoke_codex(prompt)
            post = self.capture_post_window_snapshot()

            gate = self.gate_step_changes(step, pre, post)
            validation_codes = gate["codes"] + self.validate_step(step, gate["changed_paths"])
            rollback = bool(validation_codes)

            if rollback:
                self.revert_window_changes(pre, post, gate["changed_paths"])

            self.record_attempt(
                AttemptResult(
                    step_name=step.name,
                    attempt=attempt,
                    variant_id=variant_id,
                    exit_code=code,
                    changed_paths=gate["changed_paths"],
                    validation_codes=validation_codes,
                    rollback=rollback,
                    retries_used=retries_used,
                    fixer_runs=0,
                    skill_used=skill["used"],
                    skill_path=skill["path"],
                    skill_hash=skill["hash"],
                    skill_excerpt_mode=skill["excerpt_mode"],
                    prompt_epoch_id=prompt_epoch_id,
                ),
                stdout=out,
                stderr=err,
            )

            self.update_policy_after_attempt(
                step=step,
                variant_id=variant_id,
                prompt_epoch_id=prompt_epoch_id,
                passed=(not validation_codes and code == 0),
                clean_pass=(not validation_codes and code == 0 and retries_used == 0),
                error_codes=validation_codes or ([f"EXIT_{code}"] if code != 0 else []),
            )

            if code == 0 and not validation_codes:
                self.log(f"step_pass={step.name} attempt={attempt}")
                self.append_prompt_map(step, variant_id, prompt_epoch_id, skill)
                return True

        return False

    def invoke_codex(self, prompt: str) -> Tuple[int, str, str, int]:
        cmd = ["codex", "exec", "--sandbox", "workspace-write", "-"]
        if self.flag_support.get("experimental_json", False):
            cmd.insert(2, "--experimental-json")
        elif "--experimental-json" in cmd:
            raise RuntimeError(ERROR_CODEX_FLAG_UNSUPPORTED)

        retries = 0
        while retries <= 2:
            proc = subprocess.run(
                cmd,
                input=prompt,
                cwd=str(self.repo_root),
                text=True,
                capture_output=True,
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            blob = f"{stdout}\n{stderr}".lower()
            if any(x in blob for x in TRANSIENT_PATTERNS) and retries < 2:
                retries += 1
                time.sleep(0.4 * (retries + 1))
                continue
            return proc.returncode, stdout, stderr, retries

        return 1, "", ERROR_TRANSIENT_TRANSPORT, retries

    # ----------------------------- prompt assembly -------------------------

    def build_step_prompt(self, step: StepConfig, variant_text: str, skill: Dict[str, Any]) -> str:
        brief_excerpt = self.read_project_brief_excerpt()
        manifest = self.hash_manifest_excerpt(step)
        project_type = self.project_config.get("project_type", "")
        skill_block = ""
        if skill["used"]:
            skill_block = f"\n\n## Role skill instructions\n{skill['content']}"

        return textwrap.dedent(
            f"""
            You are the {step.role} specialist.

            Follow all constraints exactly:
            - Only modify allowlisted files for this step.
            - Do not modify /.orchestrator/**, .git/**, locked briefs, or forbidden paths.
            - Do not contradict the project brief.

            Step name: {step.name}
            Allowlisted paths: {json.dumps(step.allowlist)}
            project_type: {project_type}

            ## Project brief (Layer 0-2 reference)
            {brief_excerpt}

            ## Read-only workspace manifest excerpt
            {manifest}

            ## Role prompt variant
            {variant_text}
            {skill_block}
            """
        ).strip() + "\n"

    def read_project_brief_excerpt(self) -> str:
        if not self.project_brief_path.exists():
            return "PROJECT_BRIEF.md is missing."
        content = self.project_brief_path.read_text(encoding="utf-8")
        if len(content) <= 6000:
            return content
        return content[:6000] + "\n... [TRUNCATED DETERMINISTICALLY]\n"

    def hash_manifest_excerpt(self, step: StepConfig) -> str:
        hashes: List[Dict[str, str]] = []
        for rel in sorted(self.list_repo_files()):
            if rel.startswith(".git/"):
                continue
            p = self.repo_root / rel
            if p.is_file() and not p.is_symlink():
                hashes.append({"path": rel, "sha256": self.sha256_file(p)})
            if len(hashes) >= 40:
                break
        return json.dumps(hashes, indent=2)

    def resolve_skill(self, step: StepConfig) -> Dict[str, Any]:
        key = self.normalize_role_key(step.role)
        skill_path = self.repo_root / ".codex" / "skills" / key / "SKILL.md"
        if not skill_path.exists():
            return {
                "used": False,
                "path": str(skill_path.relative_to(self.repo_root)),
                "hash": "NO_SKILL",
                "excerpt_mode": "full",
                "content": "",
            }

        raw = skill_path.read_text(encoding="utf-8")
        body = self.parse_skill_body(raw)
        excerpt_mode = "full"
        if len(body) > 8000:
            body = body[:8000] + "\n[SKILL CONTENT TRUNCATED DETERMINISTICALLY]\n"
            excerpt_mode = "truncated"

        return {
            "used": True,
            "path": str(skill_path.relative_to(self.repo_root)),
            "hash": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "excerpt_mode": excerpt_mode,
            "content": body,
        }

    def parse_skill_body(self, raw: str) -> str:
        if raw.startswith("---"):
            parts = raw.split("\n---\n", 1)
            if len(parts) == 2:
                return parts[1].strip()
        return raw.strip()

    def select_variant(self, step: StepConfig) -> Tuple[str, str, str]:
        role_key = self.normalize_role_key(step.role)
        variants = self.load_variants_for_role(role_key)
        if not variants:
            variants = [{"id": "default", "content": self.default_variant_for_role(step.role), "source": "internal"}]

        skill = self.resolve_skill(step)
        prompt_epoch_id = self.compute_prompt_epoch_id(role_key, variants, skill["hash"])

        role_state = self.policy.setdefault("roles", {}).setdefault(role_key, {})
        epoch_state = role_state.setdefault(
            prompt_epoch_id,
            {
                "last_rr_index": -1,
                "variants": {
                    v["id"]: {"attempts": 0, "passes": 0, "clean_passes": 0, "failures": {}}
                    for v in variants
                },
                "commit": {"active": False, "target": "", "remaining": 0, "consecutive_not_clean": 0},
                "eliminated": [],
            },
        )

        for v in variants:
            epoch_state["variants"].setdefault(
                v["id"], {"attempts": 0, "passes": 0, "clean_passes": 0, "failures": {}}
            )

        ordered = sorted(v["id"] for v in variants)
        min_trials = int(self.policy.get("bootstrap_min_trials_per_variant", 3))
        bootstrap_done = all(epoch_state["variants"][vid]["attempts"] >= min_trials for vid in ordered)

        if not bootstrap_done:
            idx = (int(epoch_state.get("last_rr_index", -1)) + 1) % len(ordered)
            epoch_state["last_rr_index"] = idx
            selected_id = ordered[idx]
            self.write_policy()
            return selected_id, self.variant_content_by_id(variants, selected_id), prompt_epoch_id

        strategy = self.policy.get("selection_strategy", "ucb1")
        if strategy == "explore_then_commit":
            selected_id = self.pick_explore_then_commit(epoch_state, ordered)
        elif strategy == "rr_elimination":
            selected_id = self.pick_rr_elimination(epoch_state, ordered)
        else:
            selected_id = self.pick_ucb1(epoch_state, ordered)

        self.write_policy()
        return selected_id, self.variant_content_by_id(variants, selected_id), prompt_epoch_id

    def pick_ucb1(self, epoch_state: Dict[str, Any], ordered: List[str]) -> str:
        total_attempts = max(1, sum(epoch_state["variants"][v]["attempts"] for v in ordered))
        c = float(self.policy.get("ucb_c", 1.0))
        best = ("", -1.0)
        for vid in ordered:
            s = epoch_state["variants"][vid]
            attempts = max(1, s["attempts"])
            mean_clean = s["clean_passes"] / attempts
            score = mean_clean + c * math.sqrt(math.log(total_attempts) / attempts)
            if score > best[1] or (score == best[1] and (best[0] == "" or vid < best[0])):
                best = (vid, score)
        return best[0]

    def pick_explore_then_commit(self, epoch_state: Dict[str, Any], ordered: List[str]) -> str:
        commit = epoch_state.setdefault("commit", {})
        if commit.get("active") and commit.get("remaining", 0) > 0:
            commit["remaining"] -= 1
            return commit["target"]

        best = self.best_by_mean_clean(epoch_state, ordered)
        commit["active"] = True
        commit["target"] = best
        commit["remaining"] = int(self.policy.get("commit_window_runs", 10))
        return best

    def pick_rr_elimination(self, epoch_state: Dict[str, Any], ordered: List[str]) -> str:
        eliminated = set(epoch_state.setdefault("eliminated", []))
        active = [v for v in ordered if v not in eliminated]
        if not active:
            epoch_state["eliminated"] = []
            active = ordered[:]

        min_trials = int(self.policy.get("elim_min_trials", 6))
        min_mean_clean = float(self.policy.get("elim_min_mean_clean", 0.1))
        max_fail_rate = float(self.policy.get("elim_max_failure_rate", 0.9))

        for vid in active[:]:
            s = epoch_state["variants"][vid]
            if s["attempts"] >= min_trials:
                mean_clean = s["clean_passes"] / max(1, s["attempts"])
                failure_rate = 1 - (s["passes"] / max(1, s["attempts"]))
                if mean_clean < min_mean_clean or failure_rate > max_fail_rate:
                    eliminated.add(vid)
                    active.remove(vid)

        epoch_state["eliminated"] = sorted(eliminated)
        if not active:
            epoch_state["eliminated"] = []
            active = ordered[:]

        idx = (int(epoch_state.get("last_rr_index", -1)) + 1) % len(active)
        epoch_state["last_rr_index"] = idx
        return active[idx]

    def best_by_mean_clean(self, epoch_state: Dict[str, Any], ordered: List[str]) -> str:
        best = (ordered[0], -1.0)
        for vid in ordered:
            s = epoch_state["variants"][vid]
            mean = s["clean_passes"] / max(1, s["attempts"])
            if mean > best[1] or (mean == best[1] and vid < best[0]):
                best = (vid, mean)
        return best[0]

    def load_variants_for_role(self, role_key: str) -> List[Dict[str, str]]:
        variants: List[Dict[str, str]] = []

        if self.design_b:
            prompt_dir = self.repo_root / "prompts" / role_key
            if prompt_dir.exists():
                for p in sorted(prompt_dir.rglob("*.txt")):
                    variants.append(
                        {
                            "id": str(p.relative_to(prompt_dir)).replace("\\", "/"),
                            "content": p.read_text(encoding="utf-8"),
                            "source": str(p.relative_to(self.repo_root)).replace("\\", "/"),
                        }
                    )
                if variants:
                    return variants

        orch_prompt_dir = self.prompt_template_root / role_key
        if orch_prompt_dir.exists():
            for p in sorted(orch_prompt_dir.rglob("*.txt")):
                variants.append(
                    {
                        "id": str(p.relative_to(orch_prompt_dir)).replace("\\", "/"),
                        "content": p.read_text(encoding="utf-8"),
                        "source": str(p.relative_to(self.repo_root)).replace("\\", "/"),
                    }
                )

        if variants:
            return variants

        internal = self.internal_variants().get(role_key, [])
        return [{"id": v["id"], "content": v["content"], "source": "internal"} for v in internal]

    def internal_variants(self) -> Dict[str, List[Dict[str, str]]]:
        common = [
            {
                "id": "v1.txt",
                "content": "Produce deterministic, validator-ready output. Edit only allowlisted paths.",
            },
            {
                "id": "v2.txt",
                "content": "Prioritize exact file boundaries and explicit headings required by validators.",
            },
        ]
        return {
            "release_engineer": common,
            "planner": common,
            "requirements_analyst": common,
            "ux_designer": common,
            "frontend_dev": common,
            "backend_dev": common,
            "qa_tester": common,
            "docs_writer": common,
            "prompt_library_bootstrap": common,
        }

    def default_variant_for_role(self, role: str) -> str:
        return f"Fulfill the {role} step with deterministic deliverables and validator compliance."

    def variant_content_by_id(self, variants: List[Dict[str, str]], variant_id: str) -> str:
        for v in variants:
            if v["id"] == variant_id:
                return v["content"]
        return variants[0]["content"]

    def compute_prompt_epoch_id(self, role_key: str, variants: List[Dict[str, str]], skill_hash: str) -> str:
        h = hashlib.sha256()
        h.update(role_key.encode("utf-8"))
        for v in sorted(variants, key=lambda x: x["id"]):
            h.update(v["id"].encode("utf-8"))
            h.update(hashlib.sha256(v["content"].encode("utf-8")).hexdigest().encode("utf-8"))
        h.update(skill_hash.encode("utf-8"))
        return h.hexdigest()

    # ----------------------------- snapshots + gating ----------------------

    def capture_pre_window_snapshot(self) -> Dict[str, Any]:
        head = self.run_cmd(["git", "rev-parse", "HEAD"], check=True)["stdout"].strip()
        cached = self.run_cmd(["git", "diff", "--cached", "--name-only"], check=True)["stdout"].strip()
        if cached:
            self.fail_fast(ERROR_INVARIANT_GIT_CHANGED, "git index not clean at run start")

        backup_dir = Path(tempfile.mkdtemp(prefix="orchestrator_pre_"))
        pre_files = self.list_repo_files()
        pre_untracked = self.git_untracked_files()

        for rel in pre_files:
            src = self.repo_root / rel
            dst = backup_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_symlink():
                target = os.readlink(src)
                os.symlink(target, dst)
            elif src.is_file():
                shutil.copy2(src, dst)

        return {
            "head": head,
            "cached": cached,
            "files": pre_files,
            "hashes": self.hash_snapshot(pre_files),
            "untracked": pre_untracked,
            "backup_dir": str(backup_dir),
        }

    def capture_post_window_snapshot(self) -> Dict[str, Any]:
        return {
            "head": self.run_cmd(["git", "rev-parse", "HEAD"], check=True)["stdout"].strip(),
            "cached": self.run_cmd(["git", "diff", "--cached", "--name-only"], check=True)["stdout"].strip(),
            "files": self.list_repo_files(),
            "hashes": self.hash_snapshot(self.list_repo_files()),
            "untracked": self.git_untracked_files(),
        }

    def gate_step_changes(self, step: StepConfig, pre: Dict[str, Any], post: Dict[str, Any]) -> Dict[str, Any]:
        codes: List[str] = []

        if pre["head"] != post["head"] or post["cached"]:
            codes.append(ERROR_INVARIANT_GIT_CHANGED)

        changed_paths = sorted(self.compute_changed_paths(pre["files"], pre["hashes"], post["files"], post["hashes"]))

        if any(p.startswith(".orchestrator/") for p in changed_paths):
            codes.append(ERROR_INVARIANT_ORCH_CHANGED)

        forbidden_touched = [p for p in changed_paths if p.startswith(".git/")]
        if forbidden_touched:
            codes.append(ERROR_FORBIDDEN_PATH)

        for p in changed_paths:
            if not self.is_path_safe(p):
                codes.append(ERROR_ALLOWLIST)
                break
            if self.path_is_symlink(self.repo_root / p):
                codes.append(ERROR_ALLOWLIST)
                break

        allow_violations = [p for p in changed_paths if not self.is_allowlisted(p, step.allowlist, step.name)]
        if allow_violations:
            codes.append(ERROR_ALLOWLIST)

        limits = dict(DEFAULT_LIMITS)
        limits.update(self.policy.get("step_limits", {}).get(step.name, {}))
        changed_count = len(changed_paths)
        if changed_count > int(limits["max_changed_files"]):
            codes.append(ERROR_CAP_FILES)

        bytes_changed = self.estimate_bytes_changed(changed_paths, pre["hashes"], post["hashes"])
        if bytes_changed > int(limits["max_total_bytes_changed"]):
            codes.append(ERROR_CAP_BYTES)

        deleted = [p for p in pre["files"] if p not in post["files"]]
        if len(deleted) > int(limits["max_deleted_files"]):
            codes.append(ERROR_CAP_DELETES)

        for locked, enabled in self.locked_files.items():
            if enabled and locked in changed_paths:
                codes.append(ERROR_FORBIDDEN_PATH)

        return {"codes": sorted(set(codes)), "changed_paths": changed_paths}

    def revert_window_changes(self, pre: Dict[str, Any], post: Dict[str, Any], changed_paths: List[str]) -> None:
        backup_dir = Path(pre["backup_dir"])
        pre_set = set(pre["files"])
        post_set = set(post["files"])

        for rel in sorted(post_set - pre_set):
            target = self.repo_root / rel
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            elif target.exists() or target.is_symlink():
                target.unlink(missing_ok=True)

        for rel in sorted(pre_set):
            src = backup_dir / rel
            dst = self.repo_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_symlink():
                if dst.exists() or dst.is_symlink():
                    if dst.is_dir():
                        shutil.rmtree(dst, ignore_errors=True)
                    else:
                        dst.unlink(missing_ok=True)
                os.symlink(os.readlink(src), dst)
            elif src.is_file():
                shutil.copy2(src, dst)

        shutil.rmtree(backup_dir, ignore_errors=True)

    # ----------------------------- validators ------------------------------

    def validate_step(self, step: StepConfig, changed_paths: List[str]) -> List[str]:
        codes: List[str] = []

        codes.extend(self.validate_required_existence())

        if step.name == "planner":
            codes.extend(self.validate_planner_files())

        if self.backend_required:
            codes.extend(self.validate_infra_files())

        codes.extend(self.validate_project_brief_files())
        codes.extend(self.validate_content_rules(step, changed_paths))

        if self.design_b:
            codes.extend(self.validate_design_b_assets(step))

        return sorted(set(codes))

    def validate_required_existence(self) -> List[str]:
        codes: List[str] = []
        for f in REQUIRED_FILES:
            if not (self.repo_root / f).exists():
                codes.append(f"MISSING_{f}")
        for d in REQUIRED_DIRS:
            if not (self.repo_root / d).exists():
                codes.append(f"MISSING_DIR_{d}")
        return codes

    def validate_planner_files(self) -> List[str]:
        p = self.repo_root / ".pipeline_plan.json"
        if not p.exists():
            return ["PLAN_MISSING"]
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return ["PLAN_INVALID_JSON"]
        if not isinstance(data, dict):
            return ["PLAN_INVALID_OBJECT"]
        missing = [k for k in ("roles", "required_outputs", "dependencies") if k not in data]
        return [f"PLAN_MISSING_KEY_{k}" for k in missing]

    def validate_infra_files(self) -> List[str]:
        codes: List[str] = []
        compose = self.repo_root / "docker-compose.yml"
        env_example = self.repo_root / ".env.example"
        gitignore = self.repo_root / ".gitignore"
        if not compose.exists():
            codes.append("INFRA_DOCKER_COMPOSE_MISSING")
        if not env_example.exists():
            codes.append("INFRA_ENV_EXAMPLE_MISSING")
        if not gitignore.exists() or ".env" not in gitignore.read_text(encoding="utf-8", errors="ignore"):
            codes.append("INFRA_GITIGNORE_ENV_MISSING")
        return codes

    def validate_project_brief_files(self) -> List[str]:
        codes: List[str] = []
        if not self.project_brief_path.exists():
            codes.append("BRIEF_MISSING")
        else:
            brief = self.project_brief_path.read_text(encoding="utf-8", errors="ignore")
            for heading in [
                "# Layer 0",
                "# Layer 1",
                "# Layer 2",
            ]:
                if heading not in brief:
                    codes.append(f"BRIEF_HEADING_MISSING_{heading.replace('# ', '').replace(' ', '_')}")

        if self.project_brief_yaml_path.exists():
            try:
                data = json.loads(self.project_brief_yaml_path.read_text(encoding="utf-8"))
                if not isinstance(data.get("project_type"), str):
                    codes.append(ERROR_BRIEF_YAML_INVALID)
            except Exception:
                codes.append(ERROR_BRIEF_YAML_INVALID)

        return codes

    def validate_content_rules(self, step: StepConfig, changed_paths: List[str]) -> List[str]:
        codes: List[str] = []

        req = self.repo_root / "REQUIREMENTS.md"
        if req.exists():
            t = req.read_text(encoding="utf-8", errors="ignore")
            for h in ["# Overview", "# Scope", "# Non-Goals", "# Acceptance Criteria", "# Risks"]:
                if h not in t:
                    codes.append(f"REQUIREMENTS_HEADING_MISSING_{h.replace('# ', '').replace('-', '').replace(' ', '_')}")

        test_md = self.repo_root / "TEST.md"
        if test_md.exists():
            test_t = test_md.read_text(encoding="utf-8", errors="ignore")
            for h in ["# How to run tests", "# Environments"]:
                if h not in test_t:
                    codes.append(f"TEST_HEADING_MISSING_{h.replace('# ', '').replace(' ', '_')}")
            if not re.search(r"```[\s\S]*?```", test_t):
                codes.append(ERROR_TEST_CMD_MISSING)

        if step.name in {"frontend", "backend", "docs"} and "TEST.md" in changed_paths:
            codes.append("TEST_MD_OWNERSHIP_VIOLATION")

        tasks = self.repo_root / "AGENT_TASKS.md"
        if tasks.exists():
            tt = tasks.read_text(encoding="utf-8", errors="ignore")
            if "# Agent Tasks" not in tt:
                codes.append("TASKS_HEADER_MISSING")
            for sec in ["Requirements", "Designer", "Frontend", "Backend", "QA"]:
                if re.search(rf"^##\s+{re.escape(sec)}\b", tt, flags=re.MULTILINE) is None:
                    codes.append(f"TASKS_SECTION_MISSING_{sec.upper()}")
            if "Project Brief" not in tt:
                codes.append("TASKS_PROJECT_BRIEF_REF_MISSING")
            for sec in ["Requirements", "Designer", "Frontend", "Backend", "QA"]:
                m = re.search(
                    rf"^##\s+{re.escape(sec)}\b([\s\S]*?)(?=^##\s+|\Z)",
                    tt,
                    flags=re.MULTILINE,
                )
                if m:
                    bullets = re.findall(r"^\s*-\s+", m.group(1), flags=re.MULTILINE)
                    if len(bullets) < 2:
                        codes.append(f"TASKS_TOO_FEW_BULLETS_{sec.upper()}")

        readme = self.repo_root / "README.md"
        if readme.exists():
            rt = readme.read_text(encoding="utf-8", errors="ignore")
            for key in ["Backend", "Frontend", "Tests"]:
                if key not in rt:
                    codes.append(f"README_SECTION_MISSING_{key.upper()}")

        runbook = self.repo_root / "RUNBOOK.md"
        if runbook.exists():
            rb = runbook.read_text(encoding="utf-8", errors="ignore").lower()
            if "troubleshooting" not in rb:
                codes.append("RUNBOOK_TROUBLESHOOTING_MISSING")
            if "deterministic recovery" not in rb and "recovery" not in rb:
                codes.append("RUNBOOK_RECOVERY_MISSING")

        if self.backend_required and not self.frontend_references_backend():
            codes.append(ERROR_BACKEND_UNUSED)

        return codes

    def validate_design_b_assets(self, step: StepConfig) -> List[str]:
        codes: List[str] = []
        if not (self.repo_root / "AGENTS.md").exists():
            codes.append("AGENTS_MISSING")
        for d in DESIGN_B_REQUIRED_DIRS:
            if not (self.repo_root / d).exists():
                codes.append(f"DESIGN_B_DIR_MISSING_{d.replace('/', '_')}")

        if step.name in {"prompt_library_bootstrap", "prompt_tuner"}:
            codes.extend(self.validate_prompt_and_skill_quality())

        return codes

    def validate_prompt_and_skill_quality(self) -> List[str]:
        codes: List[str] = []

        prompts_dir = self.repo_root / "prompts"
        for p in prompts_dir.rglob("*.txt") if prompts_dir.exists() else []:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            if p.stat().st_size > 64 * 1024:
                codes.append("PROMPT_FILE_TOO_LARGE")
            low = txt.lower()
            if any(fs in low for fs in FORBIDDEN_SUBSTRINGS):
                codes.append("PROMPT_FORBIDDEN_SUBSTRING")
            if "allowlisted" not in low and "validator" not in low:
                codes.append(ERROR_PROMPT_QUALITY_INSUFFICIENT)

        skills_dir = self.repo_root / ".codex" / "skills"
        for p in skills_dir.rglob("SKILL.md") if skills_dir.exists() else []:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            if p.stat().st_size > 64 * 1024:
                codes.append("SKILL_FILE_TOO_LARGE")
            if not txt.startswith("---"):
                codes.append("SKILL_FRONT_MATTER_MISSING")
                continue
            parts = txt.split("\n---\n", 1)
            if len(parts) != 2:
                codes.append("SKILL_FRONT_MATTER_INVALID")
                continue
            front, body = parts
            if "name:" not in front or "description:" not in front:
                codes.append("SKILL_REQUIRED_KEYS_MISSING")
            if not body.strip():
                codes.append("SKILL_BODY_EMPTY")
            low = txt.lower()
            if any(fs in low for fs in FORBIDDEN_SUBSTRINGS):
                codes.append("SKILL_FORBIDDEN_SUBSTRING")

        return sorted(set(codes))

    def run_product_acceptance_gate(self) -> List[str]:
        codes: List[str] = []

        if self.backend_required:
            if not self.backend_has_persistence_endpoints():
                codes.append(ERROR_PERSISTENCE_MISSING)
            if not self.frontend_references_backend_persistence():
                codes.append(ERROR_PERSISTENCE_MISSING)
            if not self.sqlite_artifacts_exist():
                codes.append(ERROR_PERSISTENCE_MISSING)

        if not self.has_e2e_persistence_test():
            codes.append(ERROR_E2E_MISSING)
        if self.looks_placeholder_only():
            codes.append(ERROR_PLACEHOLDER_IMPL)
        if not self.ux_baseline_present():
            codes.append(ERROR_UX_BASELINE_FAIL)

        if not codes:
            test_codes = self.run_test_commands()
            codes.extend(test_codes)

        return sorted(set(codes))

    def run_test_commands(self) -> List[str]:
        commands: List[str] = []
        test_md = self.repo_root / "TEST.md"
        if not test_md.exists():
            return [ERROR_TEST_CMD_MISSING]

        test_text = test_md.read_text(encoding="utf-8", errors="ignore")

        tests_config = self.project_config.get("tests", {}) if isinstance(self.project_config, dict) else {}
        if (
            isinstance(tests_config, dict)
            and tests_config.get("command_source") == "profile"
            and isinstance(tests_config.get("commands"), list)
        ):
            commands = [str(c).strip() for c in tests_config["commands"] if str(c).strip()]
            for c in commands:
                if c not in test_text:
                    return ["TEST_PROFILE_NOT_DOCUMENTED"]
        else:
            m = re.search(
                r"#\s*How to run tests[\s\S]*?```(?:bash|sh|zsh|)\n([\s\S]*?)```",
                test_text,
                flags=re.IGNORECASE,
            )
            if not m:
                return [ERROR_TEST_CMD_MISSING]
            for line in m.group(1).splitlines():
                cmd = line.strip()
                if not cmd or cmd.startswith("#"):
                    continue
                commands.append(cmd)

        if not commands:
            return [ERROR_TEST_CMD_MISSING]

        for cmd in commands:
            rc = subprocess.run(cmd, cwd=str(self.repo_root), shell=True).returncode
            if rc != 0:
                return [f"TEST_CMD_FAILED_{rc}"]

        return []

    # ----------------------------- acceptance helpers ----------------------

    def backend_has_persistence_endpoints(self) -> bool:
        required = [
            "GET /api/progress",
            "POST /api/progress",
            "GET /api/scores/:levelId",
            "POST /api/scores",
        ]
        all_text = self.read_tree_text(self.repo_root / "backend")
        return all(x in all_text for x in required)

    def frontend_references_backend(self) -> bool:
        txt = self.read_tree_text(self.repo_root / "frontend")
        return "/api/levels" in txt or "GET /api/levels" in txt

    def frontend_references_backend_persistence(self) -> bool:
        txt = self.read_tree_text(self.repo_root / "frontend")
        return "/api/progress" in txt and "/api/scores" in txt

    def sqlite_artifacts_exist(self) -> bool:
        schema = self.repo_root / "backend" / "prisma" / "schema.prisma"
        migrations_dir = self.repo_root / "backend" / "prisma" / "migrations"
        return schema.exists() and migrations_dir.exists() and any(migrations_dir.rglob("*"))

    def has_e2e_persistence_test(self) -> bool:
        txt = self.read_tree_text(self.repo_root / "tests")
        needles = ["progress", "score", "reload", "persist"]
        return all(n in txt.lower() for n in needles)

    def looks_placeholder_only(self) -> bool:
        txt = self.read_tree_text(self.repo_root / "frontend") + "\n" + self.read_tree_text(self.repo_root / "backend")
        low = txt.lower()
        placeholder_hits = sum(low.count(k) for k in ["todo", "placeholder", "scaffold"]) 
        substantive_hits = sum(low.count(k) for k in ["sqlite", "prisma", "progress", "score", "jump", "move"])
        return placeholder_hits > 8 and substantive_hits < 6

    def ux_baseline_present(self) -> bool:
        txt = self.read_tree_text(self.repo_root / "frontend").lower()
        has_responsive = "@media" in txt or "sm:" in txt or "md:" in txt
        has_loading = "loading" in txt
        has_error = "error" in txt
        has_empty = "empty" in txt or "no data" in txt
        has_focus = "focus-visible" in txt or "outline" in txt
        has_tokens = "--" in txt and "color" in txt
        return all([has_responsive, has_loading, has_error, has_empty, has_focus, has_tokens])

    # ----------------------------- policy + logging ------------------------

    def update_policy_after_attempt(
        self,
        step: StepConfig,
        variant_id: str,
        prompt_epoch_id: str,
        passed: bool,
        clean_pass: bool,
        error_codes: List[str],
    ) -> None:
        role_key = self.normalize_role_key(step.role)
        role = self.policy.setdefault("roles", {}).setdefault(role_key, {})
        epoch = role.setdefault(prompt_epoch_id, {"variants": {}})
        stats = epoch.setdefault("variants", {}).setdefault(
            variant_id,
            {"attempts": 0, "passes": 0, "clean_passes": 0, "failures": {}},
        )
        stats["attempts"] += 1
        if passed:
            stats["passes"] += 1
        if clean_pass:
            stats["clean_passes"] += 1
        if not passed:
            for code in error_codes:
                stats.setdefault("failures", {})[code] = stats.setdefault("failures", {}).get(code, 0) + 1

            cp = self.policy.setdefault("constraint_patches", {}).setdefault(step.name, [])
            for code in sorted(set(error_codes)):
                line = f"if {code}: tighten constraints for {step.name}"
                if line not in cp and len(cp) < 8:
                    cp.append(line)

        self.write_policy()

    def record_attempt(self, result: AttemptResult, stdout: str, stderr: str) -> None:
        step_dir = self.run_root / "steps" / result.step_name
        step_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "step_name": result.step_name,
            "attempt": result.attempt,
            "variant_id": result.variant_id,
            "exit_code": result.exit_code,
            "changed_paths": result.changed_paths,
            "validation_codes": result.validation_codes,
            "rollback": result.rollback,
            "retries_used": result.retries_used,
            "fixer_runs": result.fixer_runs,
            "skill_used": result.skill_used,
            "skill_path": result.skill_path,
            "skill_hash": result.skill_hash,
            "skill_excerpt_mode": result.skill_excerpt_mode,
            "prompt_epoch_id": result.prompt_epoch_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.write_json(step_dir / f"attempt_{result.attempt}.json", payload)

        if self.mode == "verbose":
            (step_dir / f"attempt_{result.attempt}.stdout.log").write_text(stdout, encoding="utf-8")
            (step_dir / f"attempt_{result.attempt}.stderr.log").write_text(stderr, encoding="utf-8")

    def append_prompt_map(self, step: StepConfig, variant_id: str, prompt_epoch_id: str, skill: Dict[str, Any]) -> None:
        role_key = self.normalize_role_key(step.role)
        source = self.resolve_variant_source(role_key, variant_id)
        self.prompt_map["steps"].append(
            {
                "step": step.name,
                "agent_role": step.role,
                "variant_id": variant_id,
                "variant_source": source,
                "prompt_epoch_id": prompt_epoch_id,
                "skill_path": skill["path"],
                "skill_hash": skill["hash"],
                "skill_used": skill["used"],
                "skill_excerpt_mode": skill["excerpt_mode"],
            }
        )

    def resolve_variant_source(self, role_key: str, variant_id: str) -> str:
        variants = self.load_variants_for_role(role_key)
        for v in variants:
            if v["id"] == variant_id:
                return v.get("source", "internal")
        return "internal"

    def flush_prompt_map(self) -> None:
        self.write_json(self.run_root / "prompt_map.json", self.prompt_map)

    def write_eval(self, score: int, details: Dict[str, Any]) -> None:
        payload = {
            "run_id": self.run_id,
            "score": score,
            "details": details,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.write_json(self.evals_root / f"{self.run_id}.json", payload)

    def compute_score(self, total_retries: int, fixer_runs: int) -> int:
        score = 0
        if self.validate_required_existence() == [] and (not self.design_b or self.validate_design_b_assets(StepConfig("", "", [])) == []):
            score += 10
        if not self.validate_content_rules(StepConfig("", "", []), []):
            score += 20
        if not self.run_test_commands():
            score += 40
        if not self.run_product_acceptance_gate():
            score += 30
        score -= max(0, total_retries) * 5
        score -= max(0, fixer_runs) * 10

        changed_total = len(self.compute_changed_paths([], {}, self.list_repo_files(), self.hash_snapshot(self.list_repo_files())))
        score -= max(0, changed_total - 20)

        return max(0, score)

    # ----------------------------- utilities -------------------------------

    def prompt_library_missing_or_empty(self) -> bool:
        prompts = self.repo_root / "prompts"
        skills = self.repo_root / ".codex" / "skills"
        prompts_empty = (not prompts.exists()) or (not any(prompts.rglob("*.txt")))
        skills_empty = (not skills.exists()) or (not any(skills.rglob("SKILL.md")))
        return prompts_empty or skills_empty

    def normalize_role_key(self, role: str) -> str:
        x = role.lower()
        x = re.sub(r"[^a-z0-9]+", "_", x).strip("_")
        return x

    def run_cmd(self, cmd: List[str], check: bool = False) -> Dict[str, Any]:
        proc = subprocess.run(cmd, cwd=str(self.repo_root), capture_output=True, text=True)
        if check and proc.returncode != 0:
            raise RuntimeError(f"command_failed: {' '.join(cmd)}\n{proc.stderr}")
        return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}

    def sha256_file(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def list_repo_files(self) -> List[str]:
        out: List[str] = []
        for p in self.repo_root.rglob("*"):
            if ".git" in p.parts:
                continue
            if p.is_file() or p.is_symlink():
                out.append(str(p.relative_to(self.repo_root)).replace("\\", "/"))
        return sorted(out)

    def git_untracked_files(self) -> List[str]:
        res = self.run_cmd(["git", "ls-files", "--others", "--exclude-standard"], check=False)
        return sorted([x.strip() for x in res["stdout"].splitlines() if x.strip()])

    def hash_snapshot(self, files: List[str]) -> Dict[str, str]:
        hs: Dict[str, str] = {}
        for rel in files:
            p = self.repo_root / rel
            if p.exists() and p.is_file() and not p.is_symlink():
                hs[rel] = self.sha256_file(p)
            elif p.is_symlink():
                hs[rel] = f"symlink:{os.readlink(p)}"
            else:
                hs[rel] = "MISSING"
        return hs

    def compute_changed_paths(
        self,
        pre_files: List[str],
        pre_hashes: Dict[str, str],
        post_files: List[str],
        post_hashes: Dict[str, str],
    ) -> List[str]:
        changed: List[str] = []
        all_paths = set(pre_files) | set(post_files)
        for p in all_paths:
            if pre_hashes.get(p) != post_hashes.get(p):
                changed.append(p)
        return changed

    def estimate_bytes_changed(
        self, changed_paths: List[str], pre_hashes: Dict[str, str], post_hashes: Dict[str, str]
    ) -> int:
        total = 0
        for rel in changed_paths:
            p = self.repo_root / rel
            if p.exists() and p.is_file():
                total += p.stat().st_size
            else:
                total += 0
        return total

    def is_path_safe(self, rel_path: str) -> bool:
        normalized = Path(rel_path)
        if ".." in normalized.parts:
            return False
        resolved = (self.repo_root / rel_path).resolve(strict=False)
        try:
            resolved.relative_to(self.repo_root.resolve())
        except ValueError:
            return False
        return True

    def path_is_symlink(self, p: Path) -> bool:
        try:
            return p.is_symlink()
        except OSError:
            return True

    def is_allowlisted(self, rel_path: str, patterns: List[str], step_name: str) -> bool:
        if rel_path.startswith(".orchestrator/") or rel_path.startswith(".git/"):
            return False

        if self.design_b and step_name not in {"prompt_library_bootstrap", "prompt_tuner"}:
            if rel_path.startswith("prompts/") or rel_path.startswith(".codex/skills/"):
                return False
        if self.design_b and step_name != "release_engineer" and rel_path == "AGENTS.md":
            return False

        for pat in patterns:
            if pat.endswith("/**"):
                prefix = pat[:-3].rstrip("/") + "/"
                if rel_path.startswith(prefix):
                    return True
            elif rel_path == pat:
                return True
        return False

    def read_tree_text(self, root: Path) -> str:
        if not root.exists():
            return ""
        chunks: List[str] = []
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in {
                ".ts",
                ".tsx",
                ".js",
                ".jsx",
                ".json",
                ".md",
                ".css",
                ".prisma",
                ".sql",
                ".yml",
                ".yaml",
            }:
                try:
                    chunks.append(p.read_text(encoding="utf-8", errors="ignore"))
                except OSError:
                    pass
                if sum(len(c) for c in chunks) > 300_000:
                    break
        return "\n".join(chunks)

    def write_policy(self) -> None:
        self.write_json(self.policy_path, self.policy)

    def write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def log(self, message: str) -> None:
        if self.mode == "verbose":
            print(message, flush=True)

    def fail_fast(self, code: str, detail: str) -> None:
        print(f"{code}: {detail}", file=sys.stderr)
        raise SystemExit(2)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic Codex pipeline orchestrator")
    parser.add_argument("--design-b", action="store_true", help="Enable Design B extensions")
    parser.add_argument("--mode", choices=["quiet", "verbose"], default="verbose")
    parser.add_argument("--repo-root", default=".", help="Repository root path")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    orch = Orchestrator(repo_root=repo_root, design_b=args.design_b, mode=args.mode)
    return orch.run()


if __name__ == "__main__":
    raise SystemExit(main())

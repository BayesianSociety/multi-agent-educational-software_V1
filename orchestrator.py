#!/usr/bin/env python3
"""
Deterministic Codex pipeline orchestrator.
Design A core + Design B optional extension.
"""

from __future__ import annotations

import argparse
import copy
import fnmatch
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------- Constants ----------------------------

ERR_FLAG_UNSUPPORTED = "CODEX_CLI_FLAG_UNSUPPORTED"
ERR_NOT_GIT = "NOT_GIT_REPO"
ERR_GIT_STAGED_NOT_EMPTY = "GIT_STAGED_NOT_EMPTY"
ERR_FORBIDDEN_PATH = "FORBIDDEN_PATH_TOUCHED"
ERR_ORCH_TOUCHED = "ORCHESTRATOR_TOUCHED"
ERR_DOTGIT_TOUCHED = "DOTGIT_TOUCHED"
ERR_ALLOWLIST = "ALLOWLIST_VIOLATION"
ERR_CAP_FILES = "CAP_MAX_CHANGED_FILES_EXCEEDED"
ERR_CAP_BYTES = "CAP_MAX_TOTAL_BYTES_CHANGED_EXCEEDED"
ERR_CAP_DELETES = "CAP_MAX_DELETED_FILES_EXCEEDED"
ERR_TEST_CMD_MISSING = "TEST_CMD_MISSING"
ERR_BRIEF_YAML_INVALID = "BRIEF_YAML_INVALID"
ERR_BACKEND_UNUSED = "BACKEND_UNUSED"
ERR_TRANSIENT_EXHAUSTED = "TRANSIENT_RETRY_EXHAUSTED"

TRANSIENT_PATTERNS = (
    "stream disconnected",
    "error sending request",
    "channel closed",
)

DEFAULT_MAX_ATTEMPTS_PER_STEP = 3
DEFAULT_MAX_CHANGED_FILES = 60
DEFAULT_MAX_TOTAL_BYTES_CHANGED = 500_000
DEFAULT_MAX_DELETED_FILES = 0

FORBIDDEN_SUBSTRINGS = [
    "ignore validators",
    "bypass allowlists",
    "write outside allowed paths",
    "mark step as done even if tests fail",
    "modify .orchestrator",
]

REQUIRED_FILES_A = [
    "REQUIREMENTS.md",
    "TEST.md",
    "AGENT_TASKS.md",
    "README.md",
    "RUNBOOK.md",
]
REQUIRED_DIRS_A = ["design", "frontend", "backend", "tests"]

REQUIRED_FILES_B = ["AGENTS.md"]
REQUIRED_DIRS_B = ["prompts", ".codex/skills"]

LOCKED_BRIEF_MD = "PROJECT_BRIEF.md"
LOCKED_BRIEF_YAML = "PROJECT_BRIEF.yaml"


# ---------------------------- Data models ----------------------------

@dataclass
class StepResult:
    success: bool
    exit_code: int
    error_codes: List[str] = field(default_factory=list)
    changed_paths: List[str] = field(default_factory=list)
    changed_files: int = 0
    changed_bytes: int = 0
    deleted_files: int = 0
    retries_used: int = 0
    fixer_runs: int = 0
    rollback: bool = False
    variant_id: str = ""
    prompt_epoch_id: str = ""
    skill_path: str = ""
    skill_hash: str = ""
    skill_used: bool = False
    skill_excerpt_mode: str = "full"
    variant_source: str = ""


@dataclass
class StepSpec:
    name: str
    role: str
    allowlist: List[str]
    validator_scope: str
    max_changed_files: int = DEFAULT_MAX_CHANGED_FILES
    max_total_bytes_changed: int = DEFAULT_MAX_TOTAL_BYTES_CHANGED
    max_deleted_files: int = DEFAULT_MAX_DELETED_FILES
    can_modify_test_md: bool = False


class DeterministicError(Exception):
    def __init__(self, code: str, msg: str):
        super().__init__(f"{code}: {msg}")
        self.code = code
        self.msg = msg


# ---------------------------- Orchestrator ----------------------------

class Orchestrator:
    def __init__(self, repo_root: Path, design_mode: str, verbosity: str, enable_schema_mode: bool):
        self.repo_root = repo_root.resolve()
        self.design_mode = design_mode  # A or B
        self.verbosity = verbosity
        self.enable_schema_mode = enable_schema_mode

        self.orch_dir = self.repo_root / ".orchestrator"
        self.runs_dir = self.orch_dir / "runs"
        self.evals_dir = self.orch_dir / "evals"
        self.policy_path = self.orch_dir / "policy.json"
        self.prompt_templates_dir = self.orch_dir / "prompt_templates"

        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.current_run_dir = self.runs_dir / self.run_id
        self.current_steps_dir = self.current_run_dir / "steps"

        self.codex_features = self.detect_codex_features()

        self.brief_md_path = self.repo_root / LOCKED_BRIEF_MD
        self.brief_yaml_path = self.repo_root / LOCKED_BRIEF_YAML
        self.brief_yaml = self.load_brief_yaml_optional()
        self.backend_required = self.is_backend_required()

        self.policy = self.load_policy()
        self.prompt_map: Dict[str, dict] = {}

    # ---------- Logging ----------
    def log(self, msg: str) -> None:
        if self.verbosity == "verbose":
            print(msg)

    def info(self, msg: str) -> None:
        print(msg)

    # ---------- CLI feature detection ----------
    def detect_codex_features(self) -> dict:
        out = subprocess.run(
            ["codex", "exec", "--help"],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
        )
        if out.returncode != 0:
            raise DeterministicError("CODEX_HELP_FAILED", "Unable to run `codex exec --help`")
        help_text = out.stdout + "\n" + out.stderr
        return {
            "help_text": help_text,
            "experimental_json": "--experimental-json" in help_text,
            "output_schema": "--output-schema" in help_text,
            "ask_for_approval": "--ask-for-approval" in help_text,
            "config": "--config" in help_text,
        }

    # ---------- Policy ----------
    def default_policy(self) -> dict:
        return {
            "version": 1,
            "max_attempts_per_step": DEFAULT_MAX_ATTEMPTS_PER_STEP,
            "selection": {
                "bootstrap_min_trials_per_variant": 3,
                "strategy": "ucb1",
                "ucb_c": 1.0,
                "commit_window_runs": 10,
                "elim_min_trials": 6,
                "elim_min_mean_clean": 0.1,
                "elim_max_failure_rate": 0.9,
            },
            "steps": {},
            "constraint_patches": {},
            "limits_overrides": {},
            "feature_availability": {
                "experimental_json": self.codex_features["experimental_json"],
                "output_schema": self.codex_features["output_schema"],
                "ask_for_approval": self.codex_features["ask_for_approval"],
                "config": self.codex_features["config"],
            },
        }

    def load_policy(self) -> dict:
        if self.policy_path.exists():
            with self.policy_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        return self.default_policy()

    def save_policy(self) -> None:
        self.orch_dir.mkdir(parents=True, exist_ok=True)
        with self.policy_path.open("w", encoding="utf-8") as f:
            json.dump(self.policy, f, indent=2, sort_keys=True)

    # ---------- Brief ----------
    def load_brief_yaml_optional(self) -> Optional[dict]:
        if not self.brief_yaml_path.exists():
            return None
        try:
            text = self.brief_yaml_path.read_text(encoding="utf-8")
            parsed = json.loads(text)
            if not isinstance(parsed, dict) or not isinstance(parsed.get("project_type"), str):
                raise DeterministicError(ERR_BRIEF_YAML_INVALID, "PROJECT_BRIEF.yaml missing string project_type")
            return parsed
        except json.JSONDecodeError as e:
            raise DeterministicError(ERR_BRIEF_YAML_INVALID, f"PROJECT_BRIEF.yaml invalid JSON: {e}")

    def is_backend_required(self) -> bool:
        if self.brief_yaml and isinstance(self.brief_yaml.get("backend_required"), bool):
            return bool(self.brief_yaml["backend_required"])
        if self.brief_md_path.exists():
            text = self.brief_md_path.read_text(encoding="utf-8").lower()
            return "backend required" in text or "backend: required" in text
        return False

    def brief_excerpt(self, max_chars: int = 10_000) -> str:
        if not self.brief_md_path.exists():
            return "PROJECT_BRIEF.md not found."
        text = self.brief_md_path.read_text(encoding="utf-8")
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n\n[TRUNCATED]"

    def project_type_text(self) -> str:
        if self.brief_yaml and "project_type" in self.brief_yaml:
            return str(self.brief_yaml["project_type"])
        return "unknown"

    # ---------- Steps ----------
    def pipeline_steps(self) -> List[StepSpec]:
        steps = [
            StepSpec(
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
                    LOCKED_BRIEF_MD,
                    LOCKED_BRIEF_YAML,
                    "ARCHITECTURE_PROMPTS.md",
                ] + (["AGENTS.md"] if self.design_mode == "B" else []),
                validator_scope="release",
                can_modify_test_md=True,
            ),
            StepSpec(
                name="planner",
                role="Planner",
                allowlist=[".pipeline_plan.json", ".pipeline_plan_schema.json"],
                validator_scope="planner",
            ),
            StepSpec(
                name="requirements",
                role="Requirements Analyst",
                allowlist=["REQUIREMENTS.md", "AGENT_TASKS.md"],
                validator_scope="requirements",
            ),
            StepSpec(
                name="designer",
                role="UX / Designer",
                allowlist=["design/**", "REQUIREMENTS.md"],
                validator_scope="designer",
            ),
            StepSpec(
                name="frontend",
                role="Frontend Dev",
                allowlist=["frontend/**", "tests/**"],
                validator_scope="frontend",
            ),
            StepSpec(
                name="backend",
                role="Backend Dev",
                allowlist=["backend/**", "tests/**", ".env.example", "docker-compose.yml"],
                validator_scope="backend",
            ),
            StepSpec(
                name="qa",
                role="QA Tester",
                allowlist=["tests/**", "TEST.md"],
                validator_scope="qa",
                can_modify_test_md=True,
            ),
            StepSpec(
                name="docs",
                role="Docs Writer",
                allowlist=["README.md", "RUNBOOK.md"],
                validator_scope="docs",
            ),
        ]
        return steps

    # ---------- Prompt variants and skills ----------
    def normalize_role_key(self, role: str) -> str:
        key = role.lower().replace("/", " ").replace("-", " ")
        key = re.sub(r"\s+", "_", key).strip("_")
        return key

    def embedded_variants(self, role_key: str) -> List[Tuple[str, str, str]]:
        base = {
            "release_engineer": [
                ("v1", "Bootstrap required docs/dirs and locked brief files. Follow allowlist exactly."),
                ("v2", "Create/update bootstrap artifacts with deterministic headings and run instructions only."),
            ],
            "planner": [
                ("v1", "Write .pipeline_plan.json with roles, required_outputs, dependencies as valid JSON object."),
                ("v2", "Generate deterministic plan JSON for step sequencing and artifacts."),
            ],
            "requirements_analyst": [
                ("v1", "Refine REQUIREMENTS.md and AGENT_TASKS.md to match project brief constraints."),
                ("v2", "Update requirements/acceptance criteria and agent tasks from locked brief."),
            ],
            "ux_designer": [
                ("v1", "Produce design artifacts under /design for block editor UX and level content."),
                ("v2", "Update /design with Layer 3 UX details consistent with locked brief."),
            ],
            "frontend_dev": [
                ("v1", "Implement frontend code under /frontend and relevant tests under /tests."),
                ("v2", "Implement Next.js frontend mechanics and deterministic block runner behavior."),
            ],
            "backend_dev": [
                ("v1", "Implement backend API + SQLite persistence under /backend and tests."),
                ("v2", "Implement endpoints /health, /api/levels, /api/levels/:id using TypeScript backend."),
            ],
            "qa_tester": [
                ("v1", "Create deterministic tests and test instructions only under allowed paths."),
                ("v2", "Improve offline deterministic test coverage and TEST.md command clarity."),
            ],
            "docs_writer": [
                ("v1", "Update README.md and RUNBOOK.md with exact startup, tests, and recovery notes."),
                ("v2", "Ensure docs are coherent with architecture, commands, and troubleshooting."),
            ],
            "prompt_tuner": [
                ("v1", "Tune prompts/skills only under /prompts and /.codex/skills for better deterministic pass rate."),
                ("v2", "Refine prompt templates and skill instructions under strict guardrails only."),
            ],
        }
        items = base.get(role_key, [("v1", f"Perform role responsibilities for {role_key}.")])
        return [(v_id, txt, "embedded") for v_id, txt in items]

    def load_variants(self, role: str) -> List[Tuple[str, str, str]]:
        role_key = self.normalize_role_key(role)

        if self.design_mode == "B":
            prompts_dir = self.repo_root / "prompts" / role_key
            if prompts_dir.exists() and any(prompts_dir.glob("*.txt")):
                variants: List[Tuple[str, str, str]] = []
                for p in sorted(prompts_dir.glob("*.txt"), key=lambda x: x.name):
                    variants.append((p.stem, p.read_text(encoding="utf-8"), str(p.relative_to(self.repo_root))))
                return variants

        internal_dir = self.prompt_templates_dir / role_key
        if internal_dir.exists() and any(internal_dir.glob("*.txt")):
            variants = []
            for p in sorted(internal_dir.glob("*.txt"), key=lambda x: x.name):
                variants.append((p.stem, p.read_text(encoding="utf-8"), str(p.relative_to(self.repo_root))))
            return variants

        variants = self.embedded_variants(role_key)
        return sorted(variants, key=lambda t: t[0])

    def load_skill(self, role: str) -> Tuple[bool, str, str, str]:
        role_key = self.normalize_role_key(role)
        skill_path = self.repo_root / ".codex" / "skills" / role_key / "SKILL.md"
        if not skill_path.exists():
            return (False, "", "NO_SKILL", "")

        raw = skill_path.read_text(encoding="utf-8")
        skill_hash = sha256_text(raw)
        body = raw
        excerpt_mode = "full"
        if raw.startswith("---"):
            parts = raw.split("---", 2)
            if len(parts) >= 3:
                body = parts[2].lstrip("\n")
        if len(body) > 16_000:
            body = body[:16_000] + "\n[TRUNCATED_SKILL_BODY]"
            excerpt_mode = "truncated"
        return (True, str(skill_path.relative_to(self.repo_root)), skill_hash, body)

    def prompt_epoch_id(self, role: str, variants: List[Tuple[str, str, str]], skill_hash: str) -> str:
        role_key = self.normalize_role_key(role)
        if self.design_mode == "A":
            payload = "\n".join(f"{vid}:{sha256_text(txt)}" for vid, txt, _src in sorted(variants, key=lambda x: x[0]))
        else:
            payload = "\n".join(f"{vid}:{sha256_text(txt)}" for vid, txt, _src in sorted(variants, key=lambda x: x[0]))
            payload += f"\nskill:{skill_hash if skill_hash else 'NO_SKILL'}"
        return sha256_text(f"{role_key}\n{payload}")

    def select_variant(self, role: str, epoch_id: str, variants: List[Tuple[str, str, str]]) -> Tuple[str, str, str]:
        role_key = self.normalize_role_key(role)
        step_state = self.policy.setdefault("steps", {}).setdefault(role_key, {})
        epoch_state = step_state.setdefault("epochs", {}).setdefault(epoch_id, {})

        if epoch_state.get("initialized") is not True:
            epoch_state.clear()
            epoch_state.update(
                {
                    "initialized": True,
                    "last_rr_index": -1,
                    "variant_stats": {},
                    "commit_mode": False,
                    "commit_variant": "",
                    "commit_remaining": 0,
                    "consecutive_not_clean_best": 0,
                    "eliminated": [],
                }
            )

        for vid, _txt, _src in variants:
            epoch_state["variant_stats"].setdefault(vid, {"attempts": 0, "passes": 0, "clean_passes": 0, "failures": {}})

        cfg = self.policy["selection"]
        min_trials = int(cfg["bootstrap_min_trials_per_variant"])

        def attempts(v: str) -> int:
            return int(epoch_state["variant_stats"][v]["attempts"])

        all_bootstrapped = all(attempts(v) >= min_trials for v, _t, _s in variants)

        if not all_bootstrapped:
            idx = (int(epoch_state["last_rr_index"]) + 1) % len(variants)
            epoch_state["last_rr_index"] = idx
            return variants[idx]

        strategy = cfg.get("strategy", "ucb1")
        vids = [v for v, _t, _s in variants]

        if strategy == "ucb1":
            total_attempts = sum(attempts(v) for v in vids)
            c = float(cfg.get("ucb_c", 1.0))
            scored = []
            for v in vids:
                s = epoch_state["variant_stats"][v]
                mean_clean = float(s["clean_passes"]) / max(1, int(s["attempts"]))
                bonus = c * math.sqrt(math.log(max(1, total_attempts)) / max(1, int(s["attempts"])))
                scored.append((mean_clean + bonus, v))
            scored.sort(key=lambda x: (-x[0], x[1]))
            winner = scored[0][1]
            return next(item for item in variants if item[0] == winner)

        if strategy == "explore_then_commit":
            if epoch_state.get("commit_mode") and epoch_state.get("commit_remaining", 0) > 0:
                best = epoch_state.get("commit_variant")
                epoch_state["commit_remaining"] -= 1
                return next(item for item in variants if item[0] == best)

            means = []
            for v in vids:
                s = epoch_state["variant_stats"][v]
                means.append((float(s["clean_passes"]) / max(1, int(s["attempts"])), v))
            means.sort(key=lambda x: (-x[0], x[1]))
            best = means[0][1]
            epoch_state["commit_mode"] = True
            epoch_state["commit_variant"] = best
            epoch_state["commit_remaining"] = int(cfg.get("commit_window_runs", 10))
            return next(item for item in variants if item[0] == best)

        if strategy == "rr_elimination":
            eliminated = set(epoch_state.get("eliminated", []))
            active = [item for item in variants if item[0] not in eliminated]
            if not active:
                epoch_state["eliminated"] = []
                active = variants

            idx = (int(epoch_state["last_rr_index"]) + 1) % len(active)
            epoch_state["last_rr_index"] = idx
            chosen = active[idx]

            min_trials_elim = int(cfg.get("elim_min_trials", 6))
            min_mean_clean = float(cfg.get("elim_min_mean_clean", 0.1))
            max_failure_rate = float(cfg.get("elim_max_failure_rate", 0.9))
            for v, _t, _s in active:
                st = epoch_state["variant_stats"][v]
                a = int(st["attempts"])
                if a < min_trials_elim:
                    continue
                mean_clean = float(st["clean_passes"]) / max(1, a)
                failure_rate = 1.0 - (float(st["passes"]) / max(1, a))
                if mean_clean < min_mean_clean or failure_rate > max_failure_rate:
                    eliminated.add(v)
            epoch_state["eliminated"] = sorted(eliminated)
            return chosen

        raise DeterministicError("INVALID_SELECTION_STRATEGY", f"Unsupported strategy: {strategy}")

    # ---------- Prompt assembly ----------
    def read_manifest_snapshot(self) -> str:
        files = repo_files(self.repo_root)
        lines = []
        for rel in sorted(files):
            h = file_sha256(self.repo_root / rel)
            lines.append(f"{rel} {h}")
        text = "\n".join(lines)
        if len(text) > 16_000:
            text = text[:16_000] + "\n[TRUNCATED_MANIFEST]"
        return text

    def build_step_prompt(self, step: StepSpec, variant_text: str, skill_text: str) -> str:
        brief = self.brief_excerpt()
        manifest = self.read_manifest_snapshot()
        allowlist_text = "\n".join(f"- {p}" for p in step.allowlist)

        prompt = f"""
You are the specialist role: {step.role}

Project type: {self.project_type_text()}

Locked project brief excerpt (Layer 0-2 source of truth):
{brief}

Read-only hashed manifest snapshot:
{manifest}

Selected role prompt variant:
{variant_text}

Resolved role skill instructions:
{skill_text if skill_text else '[NO_SKILL]'}

Instructions:
- Do not contradict the locked project brief.
- Modify only allowlisted paths for this step.
- Allowed paths:
{allowlist_text}
- Never modify /.orchestrator/**
- Never modify .git/**
- Do not claim completion; just make filesystem changes.
""".strip()
        return prompt

    # ---------- Git and snapshots ----------
    def require_git_repo(self) -> None:
        out = run_cmd(["git", "rev-parse", "--is-inside-work-tree"], cwd=self.repo_root)
        if out.returncode != 0 or out.stdout.strip() != "true":
            raise DeterministicError(ERR_NOT_GIT, "Must run inside a git repository")

    def git_head(self) -> str:
        out = run_cmd(["git", "rev-parse", "HEAD"], cwd=self.repo_root)
        if out.returncode != 0:
            raise DeterministicError("GIT_HEAD_FAILED", out.stderr.strip())
        return out.stdout.strip()

    def git_staged_names(self) -> List[str]:
        out = run_cmd(["git", "diff", "--cached", "--name-only"], cwd=self.repo_root)
        if out.returncode != 0:
            raise DeterministicError("GIT_DIFF_CACHED_FAILED", out.stderr.strip())
        return [line.strip() for line in out.stdout.splitlines() if line.strip()]

    def precheck_git_integrity(self) -> dict:
        head = self.git_head()
        staged = self.git_staged_names()
        if staged:
            raise DeterministicError(ERR_GIT_STAGED_NOT_EMPTY, "git diff --cached --name-only must be empty")
        index_path = self.repo_root / ".git" / "index"
        idx_sig = file_signature(index_path) if index_path.exists() else None
        return {"head": head, "staged": staged, "index_sig": idx_sig}

    def postcheck_git_integrity(self, pre: dict) -> List[str]:
        errors: List[str] = []
        if self.git_head() != pre["head"]:
            errors.append(ERR_DOTGIT_TOUCHED)
        staged = self.git_staged_names()
        if staged:
            errors.append(ERR_DOTGIT_TOUCHED)
        idx_path = self.repo_root / ".git" / "index"
        post_sig = file_signature(idx_path) if idx_path.exists() else None
        if pre.get("index_sig") != post_sig:
            errors.append(ERR_DOTGIT_TOUCHED)
        return sorted(set(errors))

    def snapshot_tracked_and_orchestrator(self) -> Dict[str, str]:
        tracked = git_tracked_files(self.repo_root)
        orch_files = walk_files(self.orch_dir)
        all_files = sorted(set(tracked) | set(orch_files))
        snap: Dict[str, str] = {}
        for rel in all_files:
            path = self.repo_root / rel
            if path.exists() and path.is_file():
                snap[rel] = file_sha256(path)
        return snap

    def snapshot_untracked_listing(self) -> List[str]:
        out = run_cmd(["git", "ls-files", "--others", "--exclude-standard"], cwd=self.repo_root)
        if out.returncode != 0:
            raise DeterministicError("GIT_UNTRACKED_FAILED", out.stderr.strip())
        return sorted([l.strip() for l in out.stdout.splitlines() if l.strip()])

    def compute_changes(self, pre: Dict[str, str], post: Dict[str, str]) -> Tuple[List[str], List[str], List[str], int]:
        pre_keys = set(pre.keys())
        post_keys = set(post.keys())
        added = sorted(post_keys - pre_keys)
        deleted = sorted(pre_keys - post_keys)
        common = sorted(pre_keys & post_keys)
        modified = [p for p in common if pre[p] != post[p]]
        changed = sorted(set(added + deleted + modified))

        total_bytes = 0
        for rel in changed:
            p = self.repo_root / rel
            pre_size = 0
            post_size = 0
            if rel in pre_keys:
                pre_path = self.repo_root / rel
                if pre_path.exists():
                    pre_size = pre_path.stat().st_size
            if rel in post_keys and p.exists():
                post_size = p.stat().st_size
            total_bytes += abs(post_size - pre_size)
        return changed, added, deleted, total_bytes

    def revert_unauthorized(self, changed: List[str], new_untracked: List[str]) -> None:
        # Restore tracked files first
        tracked_changed = [p for p in changed if self.is_tracked(p)]
        if tracked_changed:
            run_cmd(["git", "restore", "--worktree", "--"] + tracked_changed, cwd=self.repo_root)

        # Remove unauthorized untracked paths deterministically
        for rel in sorted(new_untracked):
            path = self.repo_root / rel
            if path.is_symlink() or path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                shutil.rmtree(path, ignore_errors=True)

    def is_tracked(self, rel: str) -> bool:
        out = run_cmd(["git", "ls-files", "--error-unmatch", rel], cwd=self.repo_root)
        return out.returncode == 0

    # ---------- Path policy ----------
    def normalize_rel(self, rel: str) -> str:
        rel_path = Path(rel)
        if rel_path.is_absolute():
            raise DeterministicError(ERR_ALLOWLIST, f"Absolute path not allowed: {rel}")
        normalized = rel_path.as_posix()
        parts = Path(normalized).parts
        if ".." in parts:
            raise DeterministicError(ERR_ALLOWLIST, f"Path traversal not allowed: {rel}")
        full = (self.repo_root / rel_path).resolve()
        try:
            rel_norm = full.relative_to(self.repo_root.resolve()).as_posix()
        except ValueError:
            raise DeterministicError(ERR_ALLOWLIST, f"Path escapes repo root: {rel}")
        return rel_norm

    def match_allowlist(self, rel: str, allowlist: List[str]) -> bool:
        for pat in allowlist:
            if pat.endswith("/**"):
                base = pat[:-3]
                if rel == base or rel.startswith(base + "/"):
                    return True
            if fnmatch.fnmatch(rel, pat):
                return True
        return False

    def enforce_path_rules(
        self,
        step: StepSpec,
        changed: List[str],
        deleted: List[str],
    ) -> List[str]:
        errors: List[str] = []
        forbidden_prefixes = [".orchestrator/", ".git/"]

        for rel in changed:
            nr = self.normalize_rel(rel)

            path = self.repo_root / nr
            if path.exists() and path.is_symlink():
                errors.append(ERR_ALLOWLIST)

            if nr.startswith(".orchestrator/"):
                errors.append(ERR_ORCH_TOUCHED)
                continue
            if nr.startswith(".git/"):
                errors.append(ERR_DOTGIT_TOUCHED)
                continue
            if any(nr.startswith(pref) for pref in forbidden_prefixes):
                errors.append(ERR_FORBIDDEN_PATH)
                continue

            # Design B lock rules for prompts/skills and AGENTS.md
            if self.design_mode == "B":
                if nr.startswith("prompts/") or nr.startswith(".codex/skills/"):
                    if step.name not in {"prompt_library_bootstrap", "prompt_tuner"}:
                        errors.append(ERR_ALLOWLIST)
                if nr == "AGENTS.md" and step.name != "release_engineer":
                    errors.append(ERR_ALLOWLIST)

            # Locked brief files after step 0
            if step.name != "release_engineer" and nr in {LOCKED_BRIEF_MD, LOCKED_BRIEF_YAML}:
                errors.append(ERR_ALLOWLIST)

            if nr == "TEST.md" and not step.can_modify_test_md:
                errors.append("TEST_MD_OWNERSHIP")

            if not self.match_allowlist(nr, step.allowlist):
                errors.append(ERR_ALLOWLIST)

        if len(changed) > step.max_changed_files:
            errors.append(ERR_CAP_FILES)
        if len(deleted) > step.max_deleted_files:
            errors.append(ERR_CAP_DELETES)

        return sorted(set(errors))

    # ---------- Codex execution ----------
    def codex_cmd(self, want_experimental_json: bool = False, require_experimental_json: bool = False) -> List[str]:
        cmd = ["codex", "exec", "--sandbox", "workspace-write"]
        if want_experimental_json:
            if self.codex_features["experimental_json"]:
                cmd.append("--experimental-json")
            elif require_experimental_json:
                raise DeterministicError(ERR_FLAG_UNSUPPORTED, "--experimental-json unsupported by codex CLI")
        cmd.append("-")
        return cmd

    def run_codex_with_transient_retry(self, prompt: str, want_experimental_json: bool = False) -> Tuple[int, str, str, int]:
        retries = 2
        last_rc = 1
        last_out = ""
        last_err = ""
        for attempt in range(retries + 1):
            cmd = self.codex_cmd(want_experimental_json=want_experimental_json)
            proc = subprocess.run(
                cmd,
                cwd=self.repo_root,
                input=prompt,
                text=True,
                capture_output=True,
            )
            rc = proc.returncode
            out = proc.stdout or ""
            err = proc.stderr or ""
            low = (out + "\n" + err).lower()
            transient = any(p in low for p in TRANSIENT_PATTERNS)
            last_rc, last_out, last_err = rc, out, err
            if transient and attempt < retries:
                time.sleep(attempt + 1)
                continue
            if transient and attempt >= retries:
                return rc, out, err, retries
            return rc, out, err, attempt
        return last_rc, last_out, last_err, retries

    # ---------- Validators ----------
    def validate_required_artifacts(self) -> List[str]:
        errs: List[str] = []
        for rf in REQUIRED_FILES_A:
            if not (self.repo_root / rf).exists():
                errs.append(f"MISSING_REQUIRED_FILE:{rf}")
        for rd in REQUIRED_DIRS_A:
            if not (self.repo_root / rd).exists():
                errs.append(f"MISSING_REQUIRED_DIR:{rd}")
        if self.design_mode == "B":
            for rf in REQUIRED_FILES_B:
                if not (self.repo_root / rf).exists():
                    errs.append(f"MISSING_REQUIRED_FILE:{rf}")
            for rd in REQUIRED_DIRS_B:
                if not (self.repo_root / rd).exists():
                    errs.append(f"MISSING_REQUIRED_DIR:{rd}")
        return errs

    def validate_headings(self, rel: str, required_headings: List[str], code_prefix: str) -> List[str]:
        p = self.repo_root / rel
        if not p.exists():
            return [f"MISSING_REQUIRED_FILE:{rel}"]
        t = p.read_text(encoding="utf-8")
        errs: List[str] = []
        for h in required_headings:
            if h not in t:
                errs.append(f"{code_prefix}:{h}")
        return errs

    def validate_test_md(self) -> List[str]:
        errs = self.validate_headings("TEST.md", ["# How to run tests", "# Environments"], "TEST_MD_HEADING_MISSING")
        p = self.repo_root / "TEST.md"
        if not p.exists():
            return errs
        txt = p.read_text(encoding="utf-8")
        if "```" not in txt:
            errs.append("TEST_MD_CODEBLOCK_MISSING")
        return errs

    def validate_agent_tasks(self) -> List[str]:
        path = self.repo_root / "AGENT_TASKS.md"
        if not path.exists():
            return ["MISSING_REQUIRED_FILE:AGENT_TASKS.md"]
        txt = path.read_text(encoding="utf-8")
        errs: List[str] = []
        if "# Agent Tasks" not in txt:
            errs.append("AGENT_TASKS_HEADING_MISSING")
        for sec in ["Requirements", "Designer", "Frontend", "Backend", "QA"]:
            if f"## {sec}" not in txt:
                errs.append(f"AGENT_TASKS_SECTION_MISSING:{sec}")
        if re.search(r"Project Brief", txt, re.IGNORECASE) is None:
            errs.append("AGENT_TASKS_PROJECT_BRIEF_REF_MISSING")
        return errs

    def validate_readme_runbook(self) -> List[str]:
        errs: List[str] = []
        for rel in ["README.md", "RUNBOOK.md"]:
            p = self.repo_root / rel
            if not p.exists():
                errs.append(f"MISSING_REQUIRED_FILE:{rel}")
                continue
            t = p.read_text(encoding="utf-8").lower()
            for needed in ["frontend", "backend", "test"]:
                if needed not in t:
                    errs.append(f"{rel}_RUN_INSTR_MISSING:{needed}")
        runbook = self.repo_root / "RUNBOOK.md"
        if runbook.exists():
            t = runbook.read_text(encoding="utf-8").lower()
            if "troubleshoot" not in t and "troubleshooting" not in t:
                errs.append("RUNBOOK_TROUBLESHOOTING_MISSING")
            if "deterministic recovery" not in t:
                errs.append("RUNBOOK_RECOVERY_NOTES_MISSING")
        return errs

    def validate_project_brief(self) -> List[str]:
        errs: List[str] = []
        p = self.repo_root / LOCKED_BRIEF_MD
        if not p.exists():
            return [f"MISSING_REQUIRED_FILE:{LOCKED_BRIEF_MD}"]
        txt = p.read_text(encoding="utf-8")
        required = [
            "Layer 0",
            "Layer 1",
            "Layer 2",
            "Target platform",
            "Audience",
            "MVP",
            "Architecture constraints",
        ]
        for key in required:
            if key not in txt:
                errs.append(f"PROJECT_BRIEF_MISSING:{key}")

        if self.brief_yaml_path.exists():
            try:
                obj = json.loads(self.brief_yaml_path.read_text(encoding="utf-8"))
                if not isinstance(obj, dict) or not isinstance(obj.get("project_type"), str):
                    errs.append(ERR_BRIEF_YAML_INVALID)
            except json.JSONDecodeError:
                errs.append(ERR_BRIEF_YAML_INVALID)
        return errs

    def validate_planner(self) -> List[str]:
        errs: List[str] = []
        plan = self.repo_root / ".pipeline_plan.json"
        if not plan.exists():
            errs.append("PIPELINE_PLAN_MISSING")
            return errs
        try:
            obj = json.loads(plan.read_text(encoding="utf-8"))
            if not isinstance(obj, dict):
                errs.append("PIPELINE_PLAN_NOT_OBJECT")
            else:
                for k in ["roles", "required_outputs", "dependencies"]:
                    if k not in obj:
                        errs.append(f"PIPELINE_PLAN_KEY_MISSING:{k}")
        except json.JSONDecodeError:
            errs.append("PIPELINE_PLAN_INVALID_JSON")

        if self.enable_schema_mode:
            schema = self.repo_root / ".pipeline_plan_schema.json"
            if not schema.exists():
                errs.append("PIPELINE_PLAN_SCHEMA_MISSING")
        return errs

    def validate_infra_files_if_needed(self) -> List[str]:
        errs: List[str] = []
        if self.backend_required:
            dc = self.repo_root / "docker-compose.yml"
            env_ex = self.repo_root / ".env.example"
            gi = self.repo_root / ".gitignore"
            if not dc.exists():
                errs.append("DOCKER_COMPOSE_MISSING")
            if not env_ex.exists():
                errs.append("ENV_EXAMPLE_MISSING")
            if not gi.exists() or ".env" not in gi.read_text(encoding="utf-8"):
                errs.append("GITIGNORE_ENV_MISSING")
        return errs

    def validate_agents_md_for_b(self) -> List[str]:
        if self.design_mode != "B":
            return []
        path = self.repo_root / "AGENTS.md"
        if not path.exists():
            return ["MISSING_REQUIRED_FILE:AGENTS.md"]
        txt = path.read_text(encoding="utf-8")
        errs: List[str] = []
        for h in ["# Global Rules", "# File Boundaries", "# How to Run Tests"]:
            if h not in txt:
                errs.append(f"AGENTS_MD_HEADING_MISSING:{h}")
        if "Do not modify /.orchestrator/**" not in txt:
            errs.append("AGENTS_MD_RULE_MISSING")
        return errs

    def validate_backend_integration_evidence(self) -> List[str]:
        if not self.backend_required:
            return []
        frontend_dir = self.repo_root / "frontend"
        if not frontend_dir.exists():
            return [ERR_BACKEND_UNUSED]
        content = ""
        for p in frontend_dir.rglob("*.*"):
            if p.is_file() and p.suffix.lower() in {".ts", ".tsx", ".js", ".jsx", ".md"}:
                try:
                    content += p.read_text(encoding="utf-8", errors="ignore") + "\n"
                except OSError:
                    pass
        if "/api/levels" not in content and "api/levels" not in content:
            return [ERR_BACKEND_UNUSED]
        return []

    def extract_test_commands(self) -> List[str]:
        if self.brief_yaml:
            tests_obj = self.brief_yaml.get("tests")
            if isinstance(tests_obj, dict) and tests_obj.get("command_source") == "profile":
                cmds = tests_obj.get("commands")
                if isinstance(cmds, list) and all(isinstance(x, str) for x in cmds) and cmds:
                    # Validate TEST.md mentions these commands
                    test_md = (self.repo_root / "TEST.md").read_text(encoding="utf-8") if (self.repo_root / "TEST.md").exists() else ""
                    for c in cmds:
                        if c not in test_md:
                            raise DeterministicError(ERR_TEST_CMD_MISSING, "TEST.md missing profiled test command")
                    return cmds

        test_md_path = self.repo_root / "TEST.md"
        if not test_md_path.exists():
            raise DeterministicError(ERR_TEST_CMD_MISSING, "TEST.md missing")

        txt = test_md_path.read_text(encoding="utf-8")
        m = re.search(r"# How to run tests(?P<body>.*)", txt, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            raise DeterministicError(ERR_TEST_CMD_MISSING, "Missing # How to run tests section")
        body = m.group("body")
        cb = re.search(r"```(?:\w+)?\n(.*?)\n```", body, flags=re.DOTALL)
        if not cb:
            raise DeterministicError(ERR_TEST_CMD_MISSING, "Missing fenced code block under # How to run tests")

        lines = []
        for line in cb.group(1).splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)
        if not lines:
            raise DeterministicError(ERR_TEST_CMD_MISSING, "No executable test commands")
        return lines

    def run_test_commands(self) -> List[str]:
        errs: List[str] = []
        try:
            cmds = self.extract_test_commands()
        except DeterministicError as e:
            return [e.code]

        for c in cmds:
            proc = subprocess.run(c, cwd=self.repo_root, shell=True, text=True)
            if proc.returncode != 0:
                errs.append(f"TEST_CMD_FAILED:{c}")
                break
        return errs

    def run_validators(self, scope: str) -> List[str]:
        errs: List[str] = []

        # Core invariants
        errs.extend(self.validate_project_brief())
        errs.extend(self.validate_infra_files_if_needed())

        if scope in {"release", "requirements", "designer", "frontend", "backend", "qa", "docs", "final"}:
            errs.extend(self.validate_required_artifacts())
            errs.extend(
                self.validate_headings(
                    "REQUIREMENTS.md",
                    ["# Overview", "# Scope", "# Non-Goals", "# Acceptance Criteria", "# Risks"],
                    "REQ_HEADING_MISSING",
                )
            )
            errs.extend(self.validate_test_md())
            errs.extend(self.validate_agent_tasks())
            errs.extend(self.validate_readme_runbook())
            errs.extend(self.validate_agents_md_for_b())

        if scope == "planner":
            errs.extend(self.validate_planner())
        if scope in {"qa", "final"}:
            errs.extend(self.run_test_commands())
        if scope in {"docs", "final"}:
            errs.extend(self.validate_backend_integration_evidence())

        return sorted(set(errs))

    # ---------- Fixers ----------
    def fixer_allowlist_from_error(self, err: str) -> List[str]:
        if err.startswith("MISSING_REQUIRED_FILE:"):
            return [err.split(":", 1)[1]]
        if err.startswith("MISSING_REQUIRED_DIR:"):
            return [err.split(":", 1)[1] + "/**"]
        if err.startswith("REQ_HEADING_MISSING"):
            return ["REQUIREMENTS.md"]
        if err.startswith("TEST_MD") or err == ERR_TEST_CMD_MISSING:
            return ["TEST.md"]
        return []

    def run_fixer(self, step: StepSpec, error_code: str) -> bool:
        fix_allow = self.fixer_allowlist_from_error(error_code)
        if not fix_allow:
            return False
        fixer_step = StepSpec(
            name=f"{step.name}_fixer",
            role=step.role,
            allowlist=fix_allow,
            validator_scope=step.validator_scope,
            max_changed_files=10,
            max_total_bytes_changed=100_000,
            max_deleted_files=0,
            can_modify_test_md=step.can_modify_test_md,
        )
        prompt = (
            f"Deterministic fixer for error code {error_code}. "
            f"Only perform minimal corrections under allowlist: {', '.join(fix_allow)}. "
            "Do not touch any other file."
        )
        result = self.execute_codex_step_once(fixer_step, prompt, variant_id="fixer", epoch_id="fixer")
        return result.success

    # ---------- Single attempt execution ----------
    def execute_codex_step_once(self, step: StepSpec, prompt: str, variant_id: str, epoch_id: str) -> StepResult:
        temp_stdout = tempfile.NamedTemporaryFile(prefix="codex_out_", delete=False)
        temp_stderr = tempfile.NamedTemporaryFile(prefix="codex_err_", delete=False)
        temp_jsonl = tempfile.NamedTemporaryFile(prefix="codex_jsonl_", delete=False)
        temp_stdout.close()
        temp_stderr.close()
        temp_jsonl.close()

        pre_git = self.precheck_git_integrity()
        pre_snap = self.snapshot_tracked_and_orchestrator()
        pre_untracked = self.snapshot_untracked_listing()

        rc, out, err, transient_retries = self.run_codex_with_transient_retry(
            prompt,
            want_experimental_json=self.codex_features["experimental_json"],
        )

        Path(temp_stdout.name).write_text(out, encoding="utf-8")
        Path(temp_stderr.name).write_text(err, encoding="utf-8")
        # JSONL diagnostics may be unavailable; keep placeholder path for deterministic logging.
        Path(temp_jsonl.name).write_text("", encoding="utf-8")

        post_snap = self.snapshot_tracked_and_orchestrator()
        post_untracked = self.snapshot_untracked_listing()

        changed, added, deleted, changed_bytes = self.compute_changes(pre_snap, post_snap)

        errors = self.postcheck_git_integrity(pre_git)
        errors.extend(self.enforce_path_rules(step, changed, deleted))

        if changed_bytes > step.max_total_bytes_changed:
            errors.append(ERR_CAP_BYTES)

        new_untracked = sorted(set(post_untracked) - set(pre_untracked))
        unauthorized_new = []
        for rel in new_untracked:
            try:
                nr = self.normalize_rel(rel)
            except DeterministicError:
                unauthorized_new.append(rel)
                continue
            if not self.match_allowlist(nr, step.allowlist):
                unauthorized_new.append(rel)
            if nr.startswith(".orchestrator/") or nr.startswith(".git/"):
                unauthorized_new.append(rel)

        if unauthorized_new:
            errors.append(ERR_ALLOWLIST)

        rollback = bool(errors)
        if rollback:
            self.revert_unauthorized(changed, unauthorized_new)

        # After gating/revert, persist attempt log under .orchestrator
        self.current_steps_dir.mkdir(parents=True, exist_ok=True)
        attempt_log = {
            "step": step.name,
            "variant_id": variant_id,
            "prompt_epoch_id": epoch_id,
            "exit_code": rc,
            "changed_paths": changed,
            "changed_files": len(changed),
            "changed_bytes": changed_bytes,
            "deleted_files": len(deleted),
            "validation_codes": sorted(set(errors)),
            "rollback": rollback,
            "transient_retries": transient_retries,
            "stdout_temp": temp_stdout.name,
            "stderr_temp": temp_stderr.name,
            "jsonl_temp": temp_jsonl.name,
        }
        # caller adds attempt index

        res = StepResult(
            success=not rollback,
            exit_code=rc,
            error_codes=sorted(set(errors)),
            changed_paths=changed,
            changed_files=len(changed),
            changed_bytes=changed_bytes,
            deleted_files=len(deleted),
            retries_used=transient_retries,
            fixer_runs=0,
            rollback=rollback,
            variant_id=variant_id,
            prompt_epoch_id=epoch_id,
        )
        res._attempt_log = attempt_log  # type: ignore[attr-defined]
        return res

    # ---------- Per-step runner ----------
    def run_step(self, step: StepSpec) -> StepResult:
        role_key = self.normalize_role_key(step.role)
        variants = self.load_variants(step.role)
        skill_used, skill_path, skill_hash, skill_body = self.load_skill(step.role)
        epoch_id = self.prompt_epoch_id(step.role, variants, skill_hash)
        v_id, v_text, v_source = self.select_variant(step.role, epoch_id, variants)

        self.prompt_map[step.name] = {
            "agent_role": step.role,
            "variant_id": v_id,
            "variant_source": v_source,
            "prompt_epoch_id": epoch_id,
            "skill_path": skill_path,
            "skill_hash": skill_hash,
            "skill_used": skill_used,
            "skill_excerpt_mode": "full" if skill_used and "[TRUNCATED_SKILL_BODY]" not in skill_body else "truncated" if skill_used else "full",
        }

        prompt = self.build_step_prompt(step, v_text, skill_body)

        max_attempts = int(self.policy.get("max_attempts_per_step", DEFAULT_MAX_ATTEMPTS_PER_STEP))
        last: Optional[StepResult] = None

        for attempt in range(1, max_attempts + 1):
            self.log(f"Running step={step.name} attempt={attempt}/{max_attempts} variant={v_id}")
            result = self.execute_codex_step_once(step, prompt, variant_id=v_id, epoch_id=epoch_id)
            log_obj = copy.deepcopy(result._attempt_log)  # type: ignore[attr-defined]
            log_obj["attempt"] = attempt

            if result.success and result.exit_code == 0:
                val_errs = self.run_validators(step.validator_scope)
                if val_errs:
                    result.success = False
                    result.error_codes = sorted(set(result.error_codes + val_errs))
                    # Deterministic narrow fixer
                    fixer_ran = False
                    if attempt < max_attempts:
                        for code in val_errs:
                            if self.run_fixer(step, code):
                                fixer_ran = True
                                result.fixer_runs += 1
                                break
                    if fixer_ran:
                        # Re-validate after fixer
                        post_fix_errs = self.run_validators(step.validator_scope)
                        if not post_fix_errs:
                            result.success = True
                            result.error_codes = []
                        else:
                            result.error_codes = sorted(set(post_fix_errs))

            self.update_policy_stats(step.role, epoch_id, v_id, result)

            step_dir = self.current_steps_dir / step.name
            step_dir.mkdir(parents=True, exist_ok=True)
            with (step_dir / f"attempt_{attempt}.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        **log_obj,
                        "validation_codes": result.error_codes,
                        "success": result.success,
                        "fixer_runs": result.fixer_runs,
                    },
                    f,
                    indent=2,
                    sort_keys=True,
                )

            if result.success:
                result.variant_source = v_source
                result.skill_path = skill_path
                result.skill_hash = skill_hash
                result.skill_used = skill_used
                result.skill_excerpt_mode = self.prompt_map[step.name]["skill_excerpt_mode"]
                return result

            last = result
            # tighten constraints patch (deterministic and bounded)
            self.add_constraint_patch(step.name, result.error_codes)

        if last is None:
            raise DeterministicError("INTERNAL", f"No attempts executed for step {step.name}")
        last.variant_source = v_source
        last.skill_path = skill_path
        last.skill_hash = skill_hash
        last.skill_used = skill_used
        last.skill_excerpt_mode = self.prompt_map[step.name]["skill_excerpt_mode"]
        return last

    def add_constraint_patch(self, step_name: str, error_codes: List[str]) -> None:
        patches = self.policy.setdefault("constraint_patches", {}).setdefault(step_name, [])
        for code in sorted(set(error_codes)):
            line = f"If you hit {code}, apply minimal corrective edits only."
            if line not in patches:
                patches.append(line)
        if len(patches) > 8:
            del patches[8:]

    def update_policy_stats(self, role: str, epoch_id: str, variant_id: str, result: StepResult) -> None:
        role_key = self.normalize_role_key(role)
        state = self.policy.setdefault("steps", {}).setdefault(role_key, {}).setdefault("epochs", {}).setdefault(epoch_id, {})
        stats = state.setdefault("variant_stats", {}).setdefault(
            variant_id,
            {"attempts": 0, "passes": 0, "clean_passes": 0, "failures": {}},
        )
        stats["attempts"] = int(stats["attempts"]) + 1
        is_pass = 1 if result.success else 0
        stats["passes"] = int(stats["passes"]) + is_pass
        clean = 1 if (result.success and result.retries_used == 0 and result.fixer_runs == 0) else 0
        stats["clean_passes"] = int(stats["clean_passes"]) + clean
        if not result.success:
            for ec in result.error_codes:
                d = stats.setdefault("failures", {})
                d[ec] = int(d.get(ec, 0)) + 1

        # explore_then_commit deterministic re-explore trigger bookkeeping
        strategy = self.policy["selection"].get("strategy", "ucb1")
        if strategy == "explore_then_commit":
            if state.get("commit_variant") == variant_id:
                if clean:
                    state["consecutive_not_clean_best"] = 0
                else:
                    state["consecutive_not_clean_best"] = int(state.get("consecutive_not_clean_best", 0)) + 1
                    if state["consecutive_not_clean_best"] >= 2:
                        state["commit_mode"] = False
                        state["commit_variant"] = ""
                        state["commit_remaining"] = 0

    # ---------- Design B scoring ----------
    def deterministic_score(self, hard_invalid: bool, retries_total: int, fixer_total: int, changed_files_total: int) -> int:
        if hard_invalid:
            return -1
        score = 0
        if not self.validate_required_artifacts() and not self.validate_agents_md_for_b():
            score += 40
        content_errs = []
        content_errs.extend(self.validate_headings("REQUIREMENTS.md", ["# Overview", "# Scope", "# Non-Goals", "# Acceptance Criteria", "# Risks"], "REQ_HEADING_MISSING"))
        content_errs.extend(self.validate_test_md())
        content_errs.extend(self.validate_agent_tasks())
        content_errs.extend(self.validate_readme_runbook())
        if not content_errs:
            score += 30

        test_errs = self.run_test_commands()
        if not test_errs:
            score += 30

        score -= 5 * max(0, retries_total)
        score -= 10 * max(0, fixer_total)
        score -= max(0, changed_files_total - 20)
        return max(0, score)

    def validate_prompt_skill_guardrails(self) -> List[str]:
        errs: List[str] = []

        skills_root = self.repo_root / ".codex" / "skills"
        if skills_root.exists():
            for p in skills_root.rglob("SKILL.md"):
                if p.stat().st_size > 64 * 1024:
                    errs.append(f"SKILL_TOO_LARGE:{p.relative_to(self.repo_root)}")
                    continue
                txt = p.read_text(encoding="utf-8", errors="ignore")
                if not txt.startswith("---"):
                    errs.append(f"SKILL_FRONTMATTER_MISSING:{p.relative_to(self.repo_root)}")
                    continue
                m = re.match(r"^---\n(.*?)\n---\n?(.*)$", txt, flags=re.DOTALL)
                if not m:
                    errs.append(f"SKILL_FRONTMATTER_INVALID:{p.relative_to(self.repo_root)}")
                    continue
                front = m.group(1)
                body = m.group(2).strip()
                if "name:" not in front or "description:" not in front:
                    errs.append(f"SKILL_REQUIRED_KEYS_MISSING:{p.relative_to(self.repo_root)}")
                if not body:
                    errs.append(f"SKILL_BODY_EMPTY:{p.relative_to(self.repo_root)}")
                low = txt.lower()
                for fs in FORBIDDEN_SUBSTRINGS:
                    if fs in low:
                        errs.append(f"SKILL_FORBIDDEN_SUBSTRING:{p.relative_to(self.repo_root)}:{fs}")

        prompts_root = self.repo_root / "prompts"
        if prompts_root.exists():
            for p in prompts_root.rglob("*.txt"):
                if p.stat().st_size > 64 * 1024:
                    errs.append(f"PROMPT_TOO_LARGE:{p.relative_to(self.repo_root)}")
                    continue
                txt = p.read_text(encoding="utf-8", errors="ignore")
                low = txt.lower()
                for fs in FORBIDDEN_SUBSTRINGS:
                    if fs in low:
                        errs.append(f"PROMPT_FORBIDDEN_SUBSTRING:{p.relative_to(self.repo_root)}:{fs}")
                if "disable gating" in low or "proceed on failure" in low:
                    errs.append(f"PROMPT_GATING_BYPASS:{p.relative_to(self.repo_root)}")

        return sorted(set(errs))

    # ---------- Run ----------
    def run_pipeline(self) -> int:
        self.require_git_repo()

        self.current_run_dir.mkdir(parents=True, exist_ok=True)

        steps = self.pipeline_steps()
        step_results: Dict[str, StepResult] = {}

        retries_total = 0
        fixer_total = 0
        changed_files_total = 0

        for step in steps:
            result = self.run_step(step)
            step_results[step.name] = result
            retries_total += result.retries_used
            fixer_total += result.fixer_runs
            changed_files_total += result.changed_files

            if not result.success:
                self.save_policy()
                self.write_prompt_map()
                self.info(f"Step failed: {step.name} codes={','.join(result.error_codes)}")
                return 1

        final_errs = self.run_validators("final")
        if final_errs:
            self.save_policy()
            self.write_prompt_map()
            self.info(f"Final validation failed: {','.join(final_errs)}")
            return 1

        if self.design_mode == "B":
            score = self.deterministic_score(False, retries_total, fixer_total, changed_files_total)
            self.evals_dir.mkdir(parents=True, exist_ok=True)
            with (self.evals_dir / f"{self.run_id}.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "run_id": self.run_id,
                        "score": score,
                        "retries_total": retries_total,
                        "fixer_total": fixer_total,
                        "changed_files_total": changed_files_total,
                    },
                    f,
                    indent=2,
                    sort_keys=True,
                )

        self.save_policy()
        self.write_prompt_map()
        self.info("Pipeline completed successfully.")
        return 0

    def write_prompt_map(self) -> None:
        self.current_run_dir.mkdir(parents=True, exist_ok=True)
        with (self.current_run_dir / "prompt_map.json").open("w", encoding="utf-8") as f:
            json.dump(self.prompt_map, f, indent=2, sort_keys=True)


# ---------------------------- Utility functions ----------------------------


def run_cmd(args: List[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_signature(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    st = path.stat()
    return f"{st.st_size}:{int(st.st_mtime_ns)}"


def git_tracked_files(repo_root: Path) -> List[str]:
    out = run_cmd(["git", "ls-files"], cwd=repo_root)
    if out.returncode != 0:
        raise DeterministicError("GIT_LS_FILES_FAILED", out.stderr.strip())
    return [l.strip() for l in out.stdout.splitlines() if l.strip()]


def walk_files(root: Path) -> List[str]:
    if not root.exists():
        return []
    out: List[str] = []
    for p in root.rglob("*"):
        if p.is_file():
            out.append(p.relative_to(root.parent).as_posix())
    return out


def repo_files(repo_root: Path) -> List[str]:
    out: List[str] = []
    for p in repo_root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(repo_root).as_posix()
        if rel.startswith(".git/"):
            continue
        out.append(rel)
    return sorted(out)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Deterministic Codex orchestrator")
    ap.add_argument("--design-mode", choices=["A", "B"], default="A")
    ap.add_argument("--verbosity", choices=["quiet", "verbose"], default="verbose")
    ap.add_argument("--schema-mode", action="store_true", help="Require .pipeline_plan_schema.json in planner validation")
    return ap.parse_args(argv)


def main() -> int:
    args = parse_args()
    orch = Orchestrator(
        repo_root=Path.cwd(),
        design_mode=args.design_mode,
        verbosity=args.verbosity,
        enable_schema_mode=args.schema_mode,
    )
    try:
        return orch.run_pipeline()
    except DeterministicError as e:
        print(f"{e.code}: {e.msg}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

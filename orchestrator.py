#!/usr/bin/env python3
"""Deterministic Codex orchestrator implementing Design A with optional Design B.

This script orchestrates multiple `codex exec` subprocess runs while enforcing
strict filesystem boundaries and deterministic gating.
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
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

# ----------------------------- Constants ---------------------------------

CODEX_UNSUPPORTED_FLAG = "CODEX_CLI_FLAG_UNSUPPORTED"
BRIEF_YAML_INVALID = "BRIEF_YAML_INVALID"
TEST_CMD_MISSING = "TEST_CMD_MISSING"
BACKEND_UNUSED = "BACKEND_UNUSED"
GIT_INTEGRITY_FAIL = "GIT_INTEGRITY_FAIL"
ALLOWLIST_VIOLATION = "ALLOWLIST_VIOLATION"
FORBIDDEN_PATH_TOUCHED = "FORBIDDEN_PATH_TOUCHED"
INVARIANT_FAIL = "INVARIANT_FAIL"
CAP_EXCEEDED = "CAP_EXCEEDED"

REPO_ROOT = Path.cwd().resolve()
ORCH_ROOT = REPO_ROOT / ".orchestrator"
POLICY_PATH = ORCH_ROOT / "policy.json"
RUNS_ROOT = ORCH_ROOT / "runs"
EVALS_ROOT = ORCH_ROOT / "evals"
PROMPT_TEMPLATES_ROOT = ORCH_ROOT / "prompt_templates"
PROMPTS_ROOT = REPO_ROOT / "prompts"
SKILLS_ROOT = REPO_ROOT / ".codex" / "skills"

PROJECT_BRIEF_MD = REPO_ROOT / "PROJECT_BRIEF.md"
PROJECT_BRIEF_YAML = REPO_ROOT / "PROJECT_BRIEF.yaml"

REQUIRED_FILES = [
    "REQUIREMENTS.md",
    "TEST.md",
    "AGENT_TASKS.md",
    "README.md",
    "RUNBOOK.md",
]
REQUIRED_DIRS = ["design", "frontend", "backend", "tests"]

REQUIRED_FILES_B = ["AGENTS.md", "ARCHITECTURE_PROMPTS.md"]
REQUIRED_DIRS_B = ["prompts", ".codex/skills"]

BRIEF_REQUIRED_HEADINGS = [
    "# Layer 0",
    "# Layer 1",
    "# Layer 2",
]
BRIEF_REQUIRED_KEYWORDS = [
    "Target platform",
    "MVP",
    "Architecture constraints",
]

TEST_HEADING = "# How to run tests"
FORBIDDEN_PROMPT_SUBSTRINGS = [
    "ignore validators",
    "bypass allowlists",
    "write outside allowed paths",
    "mark step as done even if tests fail",
    "modify .orchestrator",
    "disable gating",
]

# ----------------------------- Data classes ------------------------------


@dataclass
class Caps:
    max_changed_files: int = 60
    max_total_bytes_changed: int = 500_000
    max_deleted_files: int = 0


@dataclass
class Step:
    name: str
    role: str
    allowlist: List[str]
    required_validators: List[str]
    optional: bool = False
    supports_fixer: bool = True


@dataclass
class AttemptResult:
    success: bool
    error_codes: List[str]
    exit_code: int
    changed_paths: List[str]
    deleted_paths: List[str]
    total_bytes_changed: int
    retries_used: int = 0
    fixer_runs: int = 0


@dataclass
class VariantChoice:
    variant_id: str
    variant_source: str
    prompt_epoch_id: str


@dataclass
class FeatureSupport:
    experimental_json: bool
    output_schema: bool
    ask_for_approval: bool
    config_flag: bool


# ----------------------------- Helpers -----------------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))


def normalize_rel_path(path: Path) -> str:
    rel = path.resolve().relative_to(REPO_ROOT)
    posix = rel.as_posix()
    if ".." in posix.split("/"):
        raise ValueError(f"Path traversal segment found: {posix}")
    return posix


def is_path_within_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(REPO_ROOT)
        return True
    except Exception:
        return False


def ensure_ascii_or_utf8(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def load_json_file(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def run_cmd(cmd: Sequence[str], *, input_text: Optional[str] = None, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd or REPO_ROOT),
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )


def git_cmd(args: Sequence[str]) -> subprocess.CompletedProcess:
    return run_cmd(["git", *args])


def detect_codex_support() -> Tuple[FeatureSupport, Optional[str]]:
    help_proc = run_cmd(["codex", "exec", "--help"])
    if help_proc.returncode != 0:
        return FeatureSupport(False, False, False, False), CODEX_UNSUPPORTED_FLAG
    txt = help_proc.stdout + "\n" + help_proc.stderr
    return (
        FeatureSupport(
            experimental_json=("--experimental-json" in txt),
            output_schema=("--output-schema" in txt),
            ask_for_approval=("--ask-for-approval" in txt),
            config_flag=("--config" in txt),
        ),
        None,
    )


def parse_brief_config() -> Tuple[dict, Optional[str]]:
    if not PROJECT_BRIEF_YAML.exists():
        return {}, None
    try:
        data = json.loads(PROJECT_BRIEF_YAML.read_text(encoding="utf-8"))
    except Exception:
        return {}, BRIEF_YAML_INVALID
    if not isinstance(data, dict) or not isinstance(data.get("project_type"), str):
        return {}, BRIEF_YAML_INVALID
    return data, None


def list_all_repo_files(include_untracked: bool = True) -> List[Path]:
    files: List[Path] = []
    for root, dirs, names in os.walk(REPO_ROOT):
        root_path = Path(root)
        rel_root = root_path.relative_to(REPO_ROOT).as_posix()
        if rel_root == ".git" or rel_root.startswith(".git/"):
            dirs[:] = []
            continue
        for d in list(dirs):
            d_path = root_path / d
            if d_path.is_symlink():
                files.append(d_path)
        for n in names:
            p = root_path / n
            files.append(p)
    if not include_untracked:
        tracked = git_cmd(["ls-files"]).stdout.splitlines()
        tracked_set = {str((REPO_ROOT / t).resolve()) for t in tracked}
        files = [f for f in files if str(f.resolve()) in tracked_set]
    return files


def file_sha(path: Path) -> str:
    if path.is_symlink():
        target = os.readlink(path)
        return sha256_text(f"SYMLINK:{target}")
    if path.is_dir():
        return "DIR"
    return sha256_bytes(path.read_bytes())


def build_snapshot() -> Dict[str, str]:
    snap: Dict[str, str] = {}
    for p in list_all_repo_files(include_untracked=True):
        if not is_path_within_repo(p):
            continue
        rel = p.resolve().relative_to(REPO_ROOT).as_posix()
        snap[rel] = file_sha(p)
    return snap


def untracked_listing() -> Set[str]:
    proc = git_cmd(["ls-files", "--others", "--exclude-standard"])
    return set([l.strip() for l in proc.stdout.splitlines() if l.strip()])


def changed_paths(pre: Dict[str, str], post: Dict[str, str]) -> Tuple[List[str], List[str], List[str]]:
    pre_keys = set(pre.keys())
    post_keys = set(post.keys())
    added = sorted(post_keys - pre_keys)
    deleted = sorted(pre_keys - post_keys)
    changed = sorted(k for k in (pre_keys & post_keys) if pre[k] != post[k])
    return added + changed + deleted, added, deleted


def bytes_changed_estimate(pre: Dict[str, str], post: Dict[str, str], paths: List[str]) -> int:
    total = 0
    for rel in paths:
        p = REPO_ROOT / rel
        if p.exists() and p.is_file():
            total += p.stat().st_size
    return total


def path_matches_allowlist(rel: str, allowlist: List[str]) -> bool:
    from fnmatch import fnmatch

    for pattern in allowlist:
        if pattern.endswith("/**"):
            prefix = pattern[:-3]
            if rel == prefix or rel.startswith(prefix + "/"):
                return True
        if fnmatch(rel, pattern):
            return True
    return False


def contains_forbidden(rel: str) -> bool:
    if rel == ".git" or rel.startswith(".git/"):
        return True
    if rel == ".orchestrator" or rel.startswith(".orchestrator/"):
        return True
    return False


def check_git_integrity_pre() -> Tuple[bool, Dict[str, str]]:
    head = git_cmd(["rev-parse", "HEAD"]).stdout.strip()
    staged = git_cmd(["diff", "--cached", "--name-only"]).stdout.strip()
    if staged:
        return False, {"head": head, "staged": staged}
    index = REPO_ROOT / ".git" / "index"
    idx = file_sha(index) if index.exists() else "MISSING"
    return True, {"head": head, "staged": staged, "index": idx}


def check_git_integrity_post(pre_meta: Dict[str, str]) -> Tuple[bool, str]:
    head = git_cmd(["rev-parse", "HEAD"]).stdout.strip()
    staged = git_cmd(["diff", "--cached", "--name-only"]).stdout.strip()
    idx_path = REPO_ROOT / ".git" / "index"
    idx = file_sha(idx_path) if idx_path.exists() else "MISSING"
    if head != pre_meta.get("head", ""):
        return False, "HEAD changed"
    if staged:
        return False, "staged index is not empty"
    if pre_meta.get("index") != idx:
        return False, ".git/index changed"
    return True, ""


def deterministic_revert(pre_untracked: Set[str], post_untracked: Set[str], unauthorized_new: Set[str]) -> None:
    git_cmd(["restore", "--staged", "--worktree", "--", "."])

    new_untracked = sorted(post_untracked - pre_untracked)
    to_remove = sorted(set(new_untracked) | set(unauthorized_new))
    for rel in to_remove:
        p = (REPO_ROOT / rel).resolve()
        if not is_path_within_repo(p):
            continue
        if p.exists():
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p)
            else:
                p.unlink(missing_ok=True)


def ensure_required_paths() -> None:
    for d in REQUIRED_DIRS + REQUIRED_DIRS_B:
        (REPO_ROOT / d).mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def has_heading(md: str, heading: str) -> bool:
    return heading in md


def first_fenced_block_under_heading(md: str, heading: str) -> Optional[str]:
    h_idx = md.find(heading)
    if h_idx < 0:
        return None
    part = md[h_idx + len(heading):]
    m = re.search(r"```[a-zA-Z0-9_-]*\n(.*?)\n```", part, flags=re.S)
    if not m:
        return None
    return m.group(1)


def parse_required_brief_excerpt(max_chars: int = 5000) -> str:
    text = read_text(PROJECT_BRIEF_MD)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[TRUNCATED DETERMINISTIC EXCERPT]...\n"


def manifest_hash_for_inputs(paths: List[str]) -> str:
    blobs = []
    for rel in sorted(set(paths)):
        p = REPO_ROOT / rel
        if p.exists() and p.is_file():
            blobs.append(f"{rel}:{sha256_bytes(p.read_bytes())}")
        elif p.exists() and p.is_dir():
            blobs.append(f"{rel}:DIR")
        else:
            blobs.append(f"{rel}:MISSING")
    return sha256_text("\n".join(blobs))


def default_policy() -> dict:
    return {
        "max_attempts_per_step": 3,
        "retry_transport_max": 2,
        "retry_backoff_seconds": [1, 2],
        "selection_strategy": "ucb1",
        "bootstrap_min_trials_per_variant": 3,
        "ucb_c": 1.0,
        "commit_window_runs": 10,
        "elim_min_trials": 6,
        "elim_min_mean_clean": 0.1,
        "elim_max_failure_rate": 0.9,
        "step_caps": {},
        "error_constraint_patches": {},
        "state": {},
    }


def get_caps_for_step(policy: dict, step_name: str) -> Caps:
    cfg = policy.get("step_caps", {}).get(step_name, {})
    return Caps(
        max_changed_files=min(60, int(cfg.get("max_changed_files", 60))),
        max_total_bytes_changed=min(500_000, int(cfg.get("max_total_bytes_changed", 500_000))),
        max_deleted_files=min(0, int(cfg.get("max_deleted_files", 0))),
    )


def tighten_caps(policy: dict, step_name: str) -> None:
    cur = policy.setdefault("step_caps", {}).setdefault(step_name, {})
    cur["max_changed_files"] = min(int(cur.get("max_changed_files", 60)), 40)
    cur["max_total_bytes_changed"] = min(int(cur.get("max_total_bytes_changed", 500_000)), 300_000)
    cur["max_deleted_files"] = min(int(cur.get("max_deleted_files", 0)), 0)


def record_constraint_patch(policy: dict, step_name: str, code: str, line: str) -> None:
    arr = policy.setdefault("error_constraint_patches", {}).setdefault(step_name, [])
    if line not in arr:
        arr.append(f"{code}:{line}")
    if len(arr) > 8:
        del arr[8:]


def internal_prompt_variants() -> Dict[str, Dict[str, str]]:
    base = (
        "You are the {role} specialist.\n"
        "Follow AGENTS.md rules and orchestrator allowlist.\n"
        "Use deterministic edits only.\n"
        "Do not modify forbidden paths.\n"
    )
    alt = (
        "Role: {role}.\n"
        "Complete only your assigned artifacts and nothing else.\n"
        "If uncertain, keep changes minimal and deterministic.\n"
    )
    roles = [
        "Release Engineer",
        "Planner",
        "Requirements Analyst",
        "UX / Designer",
        "Data Specialist",
        "Frontend Dev",
        "Backend Dev",
        "QA Tester",
        "Docs Writer",
        "Prompt Tuner",
    ]
    out: Dict[str, Dict[str, str]] = {}
    for r in roles:
        out[r] = {
            "v1": base.format(role=r),
            "v2": alt.format(role=r),
        }
    return out


def load_variants_for_agent(agent_role: str, design_b: bool) -> Tuple[List[Tuple[str, str, str]], str]:
    # returns list of (variant_id, content, source)
    safe_role = agent_role.lower().replace(" / ", "_").replace(" ", "_")
    variants: List[Tuple[str, str, str]] = []

    if design_b:
        role_dir = PROMPTS_ROOT / safe_role
        if role_dir.exists():
            files = sorted([p for p in role_dir.glob("*.txt") if p.is_file()])
            for p in files:
                variants.append((p.stem, p.read_text(encoding="utf-8"), p.as_posix()))
            if variants:
                return variants, "prompts"

    orch_dir = PROMPT_TEMPLATES_ROOT / safe_role
    if orch_dir.exists():
        files = sorted([p for p in orch_dir.glob("*.txt") if p.is_file()])
        for p in files:
            variants.append((p.stem, p.read_text(encoding="utf-8"), p.as_posix()))
        if variants:
            return variants, "orchestrator_templates"

    internal = internal_prompt_variants().get(agent_role, {"v1": f"Role: {agent_role}."})
    for vid in sorted(internal.keys()):
        variants.append((vid, internal[vid], "internal"))
    return variants, "internal"


def compute_prompt_epoch(agent_role: str, variants: List[Tuple[str, str, str]], design_b: bool) -> str:
    chunks = [f"{v_id}:{sha256_text(text)}" for v_id, text, _ in variants]
    if design_b:
        role_dir = SKILLS_ROOT / agent_role.lower().replace(" / ", "_").replace(" ", "_")
        if role_dir.exists():
            for p in sorted(role_dir.rglob("*")).__iter__():
                if p.is_file():
                    rel = p.relative_to(REPO_ROOT).as_posix()
                    chunks.append(f"{rel}:{sha256_bytes(p.read_bytes())}")
    return sha256_text("\n".join(chunks))


def choose_variant(policy: dict, role: str, variants: List[Tuple[str, str, str]], epoch: str) -> VariantChoice:
    state = policy.setdefault("state", {}).setdefault(role, {}).setdefault(epoch, {})
    stats = state.setdefault("stats", {})
    for vid, _, _ in variants:
        stats.setdefault(vid, {"attempts": 0, "passes": 0, "clean_passes": 0})

    sorted_ids = [v[0] for v in sorted(variants, key=lambda x: x[0])]
    bmin = int(policy.get("bootstrap_min_trials_per_variant", 3))

    in_bootstrap = any(stats[vid]["attempts"] < bmin for vid in sorted_ids)
    if in_bootstrap:
        last_rr = int(state.get("last_rr_index", -1))
        rr = (last_rr + 1) % len(sorted_ids)
        state["last_rr_index"] = rr
        chosen = sorted_ids[rr]
    else:
        strategy = policy.get("selection_strategy", "ucb1")
        if strategy == "explore_then_commit":
            chosen = select_explore_then_commit(policy, state, sorted_ids, stats)
        elif strategy == "rr_elimination":
            chosen = select_rr_elimination(policy, state, sorted_ids, stats)
        else:
            chosen = select_ucb1(policy, sorted_ids, stats)

    source = next((s for vid, _, s in variants if vid == chosen), "internal")
    return VariantChoice(variant_id=chosen, variant_source=source, prompt_epoch_id=epoch)


def select_ucb1(policy: dict, variant_ids: List[str], stats: dict) -> str:
    c = float(policy.get("ucb_c", 1.0))
    total = sum(max(1, int(stats[v]["attempts"])) for v in variant_ids)

    best = None
    best_score = -1e9
    for vid in variant_ids:
        attempts = max(1, int(stats[vid]["attempts"]))
        clean = float(stats[vid]["clean_passes"]) / float(attempts)
        score = clean + c * math.sqrt(math.log(max(1, total)) / attempts)
        if score > best_score or (abs(score - best_score) < 1e-12 and (best is None or vid < best)):
            best = vid
            best_score = score
    assert best is not None
    return best


def select_explore_then_commit(policy: dict, state: dict, variant_ids: List[str], stats: dict) -> str:
    commit_window = int(policy.get("commit_window_runs", 10))
    cmode = state.setdefault("commit_mode", {"active": False, "best": None, "remaining": 0, "consecutive_not_clean": 0})

    def mean_clean(vid: str) -> float:
        a = max(1, int(stats[vid]["attempts"]))
        return float(stats[vid]["clean_passes"]) / float(a)

    if not cmode["active"]:
        best = sorted(variant_ids, key=lambda v: (-mean_clean(v), v))[0]
        cmode["active"] = True
        cmode["best"] = best
        cmode["remaining"] = commit_window
        cmode["consecutive_not_clean"] = 0

    best = cmode["best"]
    attempts = int(stats[best]["attempts"])
    if attempts >= 10 and mean_clean(best) < 0.3:
        cmode["active"] = False
        return variant_ids[0]

    if cmode["remaining"] <= 0:
        cmode["active"] = False
        return variant_ids[0]

    cmode["remaining"] -= 1
    return best


def select_rr_elimination(policy: dict, state: dict, variant_ids: List[str], stats: dict) -> str:
    elim = state.setdefault("eliminated", [])
    elim_set = set(elim)

    min_trials = int(policy.get("elim_min_trials", 6))
    min_mean = float(policy.get("elim_min_mean_clean", 0.1))
    max_fail = float(policy.get("elim_max_failure_rate", 0.9))

    active = [v for v in variant_ids if v not in elim_set]
    for vid in active:
        a = max(1, int(stats[vid]["attempts"]))
        mean = float(stats[vid]["clean_passes"]) / float(a)
        fail_rate = 1.0 - (float(stats[vid]["passes"]) / float(a))
        if a >= min_trials and (mean < min_mean or fail_rate > max_fail):
            elim_set.add(vid)

    active = [v for v in variant_ids if v not in elim_set]
    if not active:
        elim_set.clear()
        active = variant_ids[:]

    state["eliminated"] = sorted(elim_set)
    last_rr = int(state.get("last_rr_index_rr", -1))
    rr = (last_rr + 1) % len(active)
    state["last_rr_index_rr"] = rr
    return active[rr]


def update_variant_stats(policy: dict, role: str, epoch: str, variant_id: str, passed: bool, clean_pass: bool) -> None:
    s = policy.setdefault("state", {}).setdefault(role, {}).setdefault(epoch, {}).setdefault("stats", {}).setdefault(
        variant_id, {"attempts": 0, "passes": 0, "clean_passes": 0}
    )
    s["attempts"] += 1
    if passed:
        s["passes"] += 1
    if clean_pass:
        s["clean_passes"] += 1


def validate_required_outputs(design_b: bool) -> List[str]:
    errors = []
    for f in REQUIRED_FILES:
        if not (REPO_ROOT / f).exists():
            errors.append(f"MISSING:{f}")
    for d in REQUIRED_DIRS:
        if not (REPO_ROOT / d).exists():
            errors.append(f"MISSING_DIR:{d}")
    if design_b:
        for f in REQUIRED_FILES_B:
            if not (REPO_ROOT / f).exists():
                errors.append(f"MISSING:{f}")
        for d in REQUIRED_DIRS_B:
            if not (REPO_ROOT / d).exists():
                errors.append(f"MISSING_DIR:{d}")
    return errors


def validate_brief_presence_and_content() -> List[str]:
    errs = []
    if not PROJECT_BRIEF_MD.exists():
        errs.append("BRIEF_MISSING")
        return errs
    text = read_text(PROJECT_BRIEF_MD)
    for h in BRIEF_REQUIRED_HEADINGS:
        if h not in text:
            errs.append(f"BRIEF_HEADING_MISSING:{h}")
    for kw in BRIEF_REQUIRED_KEYWORDS:
        if kw not in text:
            errs.append(f"BRIEF_KEYWORD_MISSING:{kw}")
    return errs


def validate_requirements_md() -> List[str]:
    p = REPO_ROOT / "REQUIREMENTS.md"
    md = read_text(p)
    errs = []
    for h in ["# Overview", "# Scope", "# Non-Goals", "# Acceptance Criteria", "# Risks"]:
        if h not in md:
            errs.append(f"REQ_HEADING_MISSING:{h}")
    return errs


def validate_test_md() -> List[str]:
    md = read_text(REPO_ROOT / "TEST.md")
    errs = []
    if "# How to run tests" not in md:
        errs.append("TEST_HEADING_MISSING")
    if first_fenced_block_under_heading(md, "# How to run tests") is None:
        errs.append(TEST_CMD_MISSING)
    if "# Environments" not in md:
        errs.append("TEST_ENV_MISSING")
    return errs


def validate_agent_tasks_md() -> List[str]:
    md = read_text(REPO_ROOT / "AGENT_TASKS.md")
    errs = []
    if "# Agent Tasks" not in md:
        errs.append("TASKS_HEADING_MISSING")
    for s in ["Requirements", "Designer", "Frontend", "Backend", "QA"]:
        if f"## {s}" not in md:
            errs.append(f"TASKS_SECTION_MISSING:{s}")
        section_pat = rf"## {re.escape(s)}\n(.*?)(\n## |\Z)"
        m = re.search(section_pat, md, flags=re.S)
        if m:
            bullets = len(re.findall(r"^\s*-\s+", m.group(1), flags=re.M))
            if bullets < 2:
                errs.append(f"TASKS_BULLETS_TOO_FEW:{s}")
    if "Project Brief" not in md:
        errs.append("TASKS_PROJECT_BRIEF_REF_MISSING")
    return errs


def validate_agents_md() -> List[str]:
    md = read_text(REPO_ROOT / "AGENTS.md")
    errs = []
    for h in ["# Global Rules", "# File Boundaries", "# How to Run Tests"]:
        if h not in md:
            errs.append(f"AGENTS_HEADING_MISSING:{h}")
    if "Do not modify /.orchestrator/**" not in md:
        errs.append("AGENTS_RULE_MISSING")
    return errs


def validate_readme_runbook(backend_required: bool) -> List[str]:
    errs = []
    readme = read_text(REPO_ROOT / "README.md")
    runbook = read_text(REPO_ROOT / "RUNBOOK.md")

    for phrase in ["frontend", "backend", "test"]:
        if phrase not in readme.lower():
            errs.append(f"README_RUN_MISSING:{phrase}")
    if "troubleshooting" not in runbook.lower():
        errs.append("RUNBOOK_TROUBLESHOOTING_MISSING")
    if "deterministic recovery" not in runbook.lower():
        errs.append("RUNBOOK_RECOVERY_MISSING")

    if backend_required:
        found = False
        for p in (REPO_ROOT / "frontend").rglob("*"):
            if p.is_file() and p.suffix.lower() in {".ts", ".tsx", ".js", ".jsx", ".md"}:
                txt = read_text(p)
                if "/api/levels" in txt or "GET /api/levels" in txt or "/health" in txt:
                    found = True
                    break
        if not found:
            errs.append(BACKEND_UNUSED)
    return errs


def validate_infra_files_if_backend_required(backend_required: bool) -> List[str]:
    errs = []
    if not backend_required:
        return errs
    if not (REPO_ROOT / "docker-compose.yml").exists():
        errs.append("DOCKER_COMPOSE_MISSING")
    if not (REPO_ROOT / ".env.example").exists():
        errs.append("ENV_EXAMPLE_MISSING")
    gitignore = read_text(REPO_ROOT / ".gitignore")
    if ".env" not in gitignore:
        errs.append("GITIGNORE_ENV_MISSING")
    return errs


def validate_planner_outputs(schema_mode: bool) -> List[str]:
    errs = []
    p = REPO_ROOT / ".pipeline_plan.json"
    if not p.exists():
        errs.append("PLAN_MISSING")
        return errs
    try:
        obj = json.loads(read_text(p))
    except Exception:
        errs.append("PLAN_INVALID_JSON")
        return errs
    if not isinstance(obj, dict):
        errs.append("PLAN_NOT_OBJECT")
        return errs
    for k in ["roles", "required_outputs", "dependencies"]:
        if k not in obj:
            errs.append(f"PLAN_KEY_MISSING:{k}")
    if schema_mode and not (REPO_ROOT / ".pipeline_plan_schema.json").exists():
        errs.append("PLAN_SCHEMA_MISSING")
    return errs


def validate_prompt_skill_guardrails() -> List[str]:
    errs = []
    # Prompts
    for p in PROMPTS_ROOT.rglob("*.txt"):
        if not p.is_file():
            continue
        if p.stat().st_size > 64 * 1024:
            errs.append(f"PROMPT_TOO_LARGE:{p.relative_to(REPO_ROOT).as_posix()}")
        txt = read_text(p).lower()
        for bad in FORBIDDEN_PROMPT_SUBSTRINGS:
            if bad in txt:
                errs.append(f"PROMPT_FORBIDDEN_SUBSTRING:{p.relative_to(REPO_ROOT).as_posix()}:{bad}")

    # Skills
    for sk in SKILLS_ROOT.rglob("SKILL.md"):
        if sk.stat().st_size > 64 * 1024:
            errs.append(f"SKILL_TOO_LARGE:{sk.relative_to(REPO_ROOT).as_posix()}")
        txt = read_text(sk)
        if not txt.startswith("---"):
            errs.append(f"SKILL_FRONT_MATTER_MISSING:{sk.relative_to(REPO_ROOT).as_posix()}")
        else:
            m = re.match(r"^---\n(.*?)\n---\n", txt, flags=re.S)
            if not m:
                errs.append(f"SKILL_FRONT_MATTER_INVALID:{sk.relative_to(REPO_ROOT).as_posix()}")
            else:
                fm = m.group(1)
                if not re.search(r"^name:\s*.+$", fm, flags=re.M):
                    errs.append(f"SKILL_NAME_MISSING:{sk.relative_to(REPO_ROOT).as_posix()}")
                if not re.search(r"^description:\s*.+$", fm, flags=re.M):
                    errs.append(f"SKILL_DESCRIPTION_MISSING:{sk.relative_to(REPO_ROOT).as_posix()}")
        low = txt.lower()
        for bad in FORBIDDEN_PROMPT_SUBSTRINGS:
            if bad in low:
                errs.append(f"SKILL_FORBIDDEN_SUBSTRING:{sk.relative_to(REPO_ROOT).as_posix()}:{bad}")
    return errs


def extract_test_commands(brief_cfg: dict) -> Tuple[List[str], Optional[str]]:
    md = read_text(REPO_ROOT / "TEST.md")
    tests_cfg = brief_cfg.get("tests", {}) if isinstance(brief_cfg, dict) else {}
    if isinstance(tests_cfg, dict) and tests_cfg.get("command_source") == "profile":
        cmds = tests_cfg.get("commands")
        if not isinstance(cmds, list) or not all(isinstance(x, str) for x in cmds):
            return [], TEST_CMD_MISSING
        # deterministic string check: TEST.md documents those commands
        for c in cmds:
            if c not in md:
                return [], f"TEST_PROFILE_NOT_DOCUMENTED:{c}"
        return cmds, None

    block = first_fenced_block_under_heading(md, TEST_HEADING)
    if block is None:
        return [], TEST_CMD_MISSING
    cmds = []
    for line in block.splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        cmds.append(t)
    if not cmds:
        return [], TEST_CMD_MISSING
    return cmds, None


def run_test_commands(cmds: List[str], verbose: bool) -> Tuple[bool, List[dict]]:
    logs = []
    ok = True
    for c in cmds:
        proc = run_cmd(["bash", "-lc", c])
        logs.append({"command": c, "exit_code": proc.returncode})
        if verbose:
            print(f"[tests] {c} -> {proc.returncode}")
        if proc.returncode != 0:
            ok = False
            break
    return ok, logs


def build_prompt(step: Step, variant_text: str, brief_cfg: dict, manifest_hash: str) -> str:
    brief_excerpt = parse_required_brief_excerpt()
    project_type = brief_cfg.get("project_type") if isinstance(brief_cfg, dict) else None
    project_type_line = f"project_type: {project_type}\n" if project_type else ""

    allowlist_lines = "\n".join(f"- {a}" for a in step.allowlist)
    return (
        f"{variant_text}\n"
        "Deterministic constraints:\n"
        "- Do not contradict the project brief.\n"
        "- Do not modify forbidden paths (.orchestrator/**, .git/**).\n"
        f"- Modify only allowlisted paths for this step:\n{allowlist_lines}\n"
        f"- Step name: {step.name}\n"
        f"- Role: {step.role}\n"
        f"- Read-only hashed input manifest: {manifest_hash}\n"
        f"{project_type_line}"
        "Project brief excerpt (Layer 0-2 source of truth):\n"
        f"{brief_excerpt}\n"
    )


def make_steps(backend_required: bool, design_b: bool, use_data_specialist: bool = False) -> List[Step]:
    steps = [
        Step(
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
                "AGENTS.md",
                "ARCHITECTURE_PROMPTS.md",
            ],
            required_validators=["required_outputs", "brief", "infra"],
        ),
        Step(
            name="planner",
            role="Planner",
            allowlist=[".pipeline_plan.json", ".pipeline_plan_schema.json"],
            required_validators=["planner"],
        ),
        Step(
            name="requirements",
            role="Requirements Analyst",
            allowlist=["REQUIREMENTS.md", "AGENT_TASKS.md"],
            required_validators=["requirements", "agent_tasks"],
        ),
        Step(
            name="designer",
            role="UX / Designer",
            allowlist=["design/**", "REQUIREMENTS.md"],
            required_validators=[],
        ),
    ]

    if use_data_specialist:
        steps.append(
            Step(
                name="data_specialist",
                role="Data Specialist",
                allowlist=["design/**"],
                required_validators=[],
                optional=True,
            )
        )

    steps.extend(
        [
            Step(
                name="frontend",
                role="Frontend Dev",
                allowlist=["frontend/**", "tests/**"],
                required_validators=["readme_runbook"],
            ),
            Step(
                name="backend",
                role="Backend Dev",
                allowlist=["backend/**", "tests/**", ".env.example", "docker-compose.yml"],
                required_validators=["readme_runbook"],
                optional=not backend_required,
            ),
            Step(
                name="qa",
                role="QA Tester",
                allowlist=["tests/**", "TEST.md"],
                required_validators=["test_md", "tests_exec"],
            ),
            Step(
                name="docs",
                role="Docs Writer",
                allowlist=["README.md", "RUNBOOK.md"],
                required_validators=["readme_runbook"],
            ),
        ]
    )

    if design_b:
        steps.append(
            Step(
                name="prompt_tuner",
                role="Prompt Tuner",
                allowlist=["prompts/**", ".codex/skills/**"],
                required_validators=["prompt_guardrails"],
                optional=True,
            )
        )

    return steps


def run_codex_with_retry(prompt: str, features: FeatureSupport, *, use_json: bool) -> Tuple[int, str, str, List[str]]:
    retries = 2
    backoff = [1, 2]
    transport_markers = ["stream disconnected", "error sending request", "channel closed"]
    attempts_logs = []

    for i in range(retries + 1):
        cmd = ["codex", "exec", "--sandbox", "workspace-write"]
        if use_json and features.experimental_json:
            cmd.append("--experimental-json")
        cmd.append("-")

        proc = run_cmd(cmd, input_text=prompt)
        out = proc.stdout
        err = proc.stderr
        attempts_logs.append(f"attempt={i+1} rc={proc.returncode}")

        combined = (out + "\n" + err).lower()
        transport_fail = any(m in combined for m in transport_markers)
        if transport_fail and i < retries:
            time.sleep(backoff[min(i, len(backoff) - 1)])
            continue
        return proc.returncode, out, err, attempts_logs

    return 1, "", "", attempts_logs


def validate_step(step: Step, backend_required: bool, design_b: bool, schema_mode: bool, brief_cfg: dict) -> List[str]:
    errs: List[str] = []
    validators = set(step.required_validators)

    if "required_outputs" in validators:
        errs.extend(validate_required_outputs(design_b=design_b))

    if "brief" in validators:
        errs.extend(validate_brief_presence_and_content())

    if "infra" in validators:
        errs.extend(validate_infra_files_if_backend_required(backend_required))

    if "planner" in validators:
        errs.extend(validate_planner_outputs(schema_mode=schema_mode))

    if "requirements" in validators:
        errs.extend(validate_requirements_md())

    if "agent_tasks" in validators:
        errs.extend(validate_agent_tasks_md())

    if "test_md" in validators:
        errs.extend(validate_test_md())

    if "readme_runbook" in validators:
        errs.extend(validate_readme_runbook(backend_required=backend_required))

    if "prompt_guardrails" in validators:
        errs.extend(validate_prompt_skill_guardrails())

    if design_b:
        errs.extend(validate_agents_md())

    if PROJECT_BRIEF_YAML.exists():
        _, cfg_err = parse_brief_config()
        if cfg_err:
            errs.append(cfg_err)

    return errs


def lock_violations(step: Step, changed: List[str], design_b: bool) -> List[str]:
    errs = []
    for rel in changed:
        # lock brief and config after step 0
        if step.name != "release_engineer":
            if rel == "PROJECT_BRIEF.md":
                errs.append("BRIEF_LOCK_VIOLATION")
            if rel == "PROJECT_BRIEF.yaml":
                errs.append("BRIEF_YAML_LOCK_VIOLATION")
        if design_b and step.name != "release_engineer" and rel == "AGENTS.md":
            errs.append("AGENTS_LOCK_VIOLATION")
    return errs


def test_md_ownership_violation(step: Step, changed: List[str]) -> List[str]:
    if "TEST.md" in changed and step.name not in {"release_engineer", "qa"}:
        return ["TEST_MD_OWNERSHIP_VIOLATION"]
    return []


def cap_violations(changed: List[str], deleted: List[str], total_bytes: int, caps: Caps) -> List[str]:
    errs = []
    if len(changed) > caps.max_changed_files:
        errs.append(f"{CAP_EXCEEDED}:CHANGED_FILES")
    if total_bytes > caps.max_total_bytes_changed:
        errs.append(f"{CAP_EXCEEDED}:BYTES")
    if len(deleted) > caps.max_deleted_files:
        errs.append(f"{CAP_EXCEEDED}:DELETIONS")
    return errs


def allowlist_violations(step: Step, changed: List[str]) -> List[str]:
    errs = []
    for rel in changed:
        p = REPO_ROOT / rel
        if p.exists() and p.is_symlink():
            errs.append(f"{ALLOWLIST_VIOLATION}:SYMLINK:{rel}")
            continue
        if contains_forbidden(rel):
            errs.append(f"{FORBIDDEN_PATH_TOUCHED}:{rel}")
            continue
        if not path_matches_allowlist(rel, step.allowlist):
            errs.append(f"{ALLOWLIST_VIOLATION}:{rel}")
    return errs


def build_attempt_record(
    *,
    run_id: str,
    step: Step,
    attempt_no: int,
    variant: VariantChoice,
    exit_code: int,
    changed: List[str],
    deleted: List[str],
    errs: List[str],
    rollback: bool,
    retries_used: int,
    fixer_runs: int,
    transport_attempts: List[str],
) -> dict:
    return {
        "run_id": run_id,
        "timestamp": now_iso(),
        "step": step.name,
        "role": step.role,
        "attempt": attempt_no,
        "variant_id": variant.variant_id,
        "variant_source": variant.variant_source,
        "prompt_epoch_id": variant.prompt_epoch_id,
        "exit_code": exit_code,
        "changed_paths": changed,
        "deleted_paths": deleted,
        "validation_codes": errs,
        "rollback": rollback,
        "retries_used": retries_used,
        "fixer_runs": fixer_runs,
        "transport_attempts": transport_attempts,
    }


def run_single_step(
    *,
    step: Step,
    run_id: str,
    policy: dict,
    features: FeatureSupport,
    backend_required: bool,
    design_b: bool,
    schema_mode: bool,
    brief_cfg: dict,
    prompt_map: dict,
    verbose: bool,
) -> AttemptResult:
    max_attempts = int(policy.get("max_attempts_per_step", 3))
    step_dir = RUNS_ROOT / run_id / "steps" / step.name
    step_dir.mkdir(parents=True, exist_ok=True)

    variants, _src = load_variants_for_agent(step.role, design_b=design_b)
    epoch = compute_prompt_epoch(step.role, variants, design_b=design_b)

    fixer_runs = 0
    retries_used = 0

    for attempt in range(1, max_attempts + 1):
        variant = choose_variant(policy, step.role, variants, epoch)
        variant_text = next(t for vid, t, _ in variants if vid == variant.variant_id)
        manifest_hash = manifest_hash_for_inputs(step.allowlist + ["PROJECT_BRIEF.md", "PROJECT_BRIEF.yaml"])
        prompt = build_prompt(step, variant_text, brief_cfg, manifest_hash)

        prompt_map.setdefault("steps", {})[step.name] = {
            "agent_role": step.role,
            "variant_id": variant.variant_id,
            "variant_source": variant.variant_source,
            "prompt_epoch_id": variant.prompt_epoch_id,
        }

        pre_ok, git_pre = check_git_integrity_pre()
        if not pre_ok:
            return AttemptResult(False, [GIT_INTEGRITY_FAIL], 1, [], [], 0, retries_used, fixer_runs)

        pre_snap = build_snapshot()
        pre_untracked = untracked_listing()

        rc, out, err, transport_attempts = run_codex_with_retry(
            prompt,
            features,
            use_json=True,
        )

        post_snap = build_snapshot()
        post_untracked = untracked_listing()

        post_ok, git_reason = check_git_integrity_post(git_pre)
        changed, _added, deleted = changed_paths(pre_snap, post_snap)
        total_bytes = bytes_changed_estimate(pre_snap, post_snap, changed)

        errs: List[str] = []

        if not post_ok:
            errs.append(f"{GIT_INTEGRITY_FAIL}:{git_reason}")

        errs.extend(lock_violations(step, changed, design_b))
        errs.extend(test_md_ownership_violation(step, changed))
        errs.extend(allowlist_violations(step, changed))
        caps = get_caps_for_step(policy, step.name)
        errs.extend(cap_violations(changed, deleted, total_bytes, caps))

        # step validators (deterministic; based on filesystem + exit/test codes)
        step_validator_errs = validate_step(
            step,
            backend_required=backend_required,
            design_b=design_b,
            schema_mode=schema_mode,
            brief_cfg=brief_cfg,
        )
        errs.extend(step_validator_errs)

        # QA step may run test commands deterministically
        if "tests_exec" in step.required_validators:
            cmds, cmd_err = extract_test_commands(brief_cfg)
            if cmd_err:
                errs.append(cmd_err)
            else:
                tests_ok, test_logs = run_test_commands(cmds, verbose=verbose)
                (step_dir / f"attempt_{attempt}_tests.json").write_text(
                    json.dumps({"commands": cmds, "logs": test_logs}, indent=2), encoding="utf-8"
                )
                if not tests_ok:
                    errs.append("TESTS_FAILED")

        # bounded fixer for simple deterministic failures
        should_fix = step.supports_fixer and any(e.startswith("MISSING:") for e in errs)
        if should_fix and fixer_runs < 1 and attempt < max_attempts:
            fixer_runs += 1
            retries_used += 1
            missing = [e.split(":", 1)[1] for e in errs if e.startswith("MISSING:")]
            f_prompt = (
                "Deterministic fixer run. Create only missing required files.\n"
                f"Allowed targets: {', '.join(missing)}\n"
                "Do not modify any other path.\n"
            )
            # fixer with minimal allowlist (same boundary checks still apply on next attempt)
            run_codex_with_retry(f_prompt, features, use_json=True)

        rollback = bool(errs)
        if rollback:
            unauthorized_new = {p for p in (post_untracked - pre_untracked) if not path_matches_allowlist(p, step.allowlist)}
            deterministic_revert(pre_untracked, post_untracked, unauthorized_new)
            tighten_caps(policy, step.name)
            for e in errs:
                record_constraint_patch(policy, step.name, e, "narrow output and path scope")
        else:
            # success path
            pass

        # persist attempt logs only after gating/revert
        attempt_record = build_attempt_record(
            run_id=run_id,
            step=step,
            attempt_no=attempt,
            variant=variant,
            exit_code=rc,
            changed=changed,
            deleted=deleted,
            errs=errs,
            rollback=rollback,
            retries_used=retries_used,
            fixer_runs=fixer_runs,
            transport_attempts=transport_attempts,
        )
        (step_dir / f"attempt_{attempt}.json").write_text(json.dumps(attempt_record, indent=2), encoding="utf-8")
        (step_dir / f"attempt_{attempt}_stdout.log").write_text(out, encoding="utf-8")
        (step_dir / f"attempt_{attempt}_stderr.log").write_text(err, encoding="utf-8")

        passed = not errs and rc == 0
        clean_pass = passed and retries_used == 0 and fixer_runs == 0
        update_variant_stats(policy, step.role, epoch, variant.variant_id, passed=passed, clean_pass=clean_pass)

        if verbose:
            print(f"[{step.name}] attempt={attempt} rc={rc} errs={len(errs)} changed={len(changed)}")

        if passed:
            return AttemptResult(True, [], rc, changed, deleted, total_bytes, retries_used, fixer_runs)

        retries_used += 1

    return AttemptResult(False, [INVARIANT_FAIL], 1, [], [], 0, retries_used, fixer_runs)


def compute_design_b_score(all_valid: bool, tests_ok: bool, retries: int, fixer_runs: int, changed_files_total: int, hard_invalid: bool) -> int:
    if hard_invalid:
        return -1
    score = 0
    if all_valid:
        score += 40
        score += 30
    if tests_ok:
        score += 30
    score -= 5 * max(0, retries)
    score -= 10 * max(0, fixer_runs)
    score -= max(0, changed_files_total - 20)
    return max(0, score)


def write_prompt_map(run_id: str, prompt_map: dict) -> None:
    p = RUNS_ROOT / run_id / "prompt_map.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(prompt_map, indent=2), encoding="utf-8")


def bootstrap_required_artifacts() -> None:
    ensure_required_paths()

    defaults = {
        "REQUIREMENTS.md": """# Overview\nEducational block-coding game for ages 7-12.\n\n# Scope\nMVP with deterministic Move/Jump execution, 10 levels, backend-backed level/progress persistence.\n\n# Non-Goals\n- Multiplayer\n- Cloud accounts\n- Advanced Scratch features\n\n# Acceptance Criteria\n- Deterministic block execution with current-block highlight\n- Backend endpoints return levels\n- SQLite persists progress\n- Tests exit 0\n\n# Risks\n- Frontend/backend drift\n- Content scale and level balancing\n""",
        "TEST.md": """# How to run tests\n```bash\ndocker compose up -d\necho \"replace with deterministic test command\"\n```\n\n# Environments\n- Local docker-compose with SQLite-backed backend service\n""",
        "AGENT_TASKS.md": """# Agent Tasks\nProject Brief: follow PROJECT_BRIEF.md Layer 0-2 constraints.\n\n## Requirements\n- Define acceptance criteria tied to Project Brief architecture constraints.\n- Keep non-goals explicit for MVP.\n\n## Designer\n- Produce deterministic level UX notes under /design.\n- Align move/jump scale to background.png.\n\n## Frontend\n- Implement block editor and deterministic runner UI.\n- Integrate calls to backend level endpoints.\n\n## Backend\n- Implement GET /health and levels endpoints.\n- Store/retrieve level/progress data in SQLite.\n\n## QA\n- Define deterministic offline test commands.\n- Validate acceptance criteria and backend integration paths.\n""",
        "RUNBOOK.md": """# Operations Runbook\n## Troubleshooting\n- Verify docker services are healthy.\n- Verify backend `/health` endpoint responds.\n\n## Deterministic Recovery\n- Re-run orchestrator step with strict allowlist/revert enabled.\n- Restore workspace with git and remove unauthorized untracked outputs.\n""",
        "ARCHITECTURE_PROMPTS.md": """# Prompt Architecture\n- Prompt variants load from `prompts/<agent>/*.txt` when present (Design B), otherwise from internal/orchestrator templates.\n- Variant selection happens in `choose_variant()` and strategy helpers (`select_ucb1`, `select_explore_then_commit`, `select_rr_elimination`).\n- Prompt assembly happens in `build_prompt()` and includes brief excerpt + hashed manifest.\n""",
        "AGENTS.md": """# Global Rules\n- Do not modify /.orchestrator/**\n- Do not modify .git/**\n- Respect per-step allowlists and deterministic validators.\n\n# File Boundaries\n- Specialist steps may edit only their explicit allowlisted paths.\n- Prompt library edits are limited to /prompts/** and /.codex/skills/** when enabled.\n\n# How to Run Tests\n- Read TEST.md and run deterministic commands exactly as documented.\n""",
        "README.md": """# multi-agent-educational-software_V1\n\n## Run frontend\n```bash\ncd frontend\nnpm run dev\n```\n\n## Run backend\n```bash\ncd backend\nnpm run dev\n```\n\n## Run tests\n```bash\n# see TEST.md for deterministic test command profile\n```\n\n## Docker compose (optional/local infra)\n```bash\ndocker compose up -d\n```\n""",
    }

    for name, content in defaults.items():
        p = REPO_ROOT / name
        if not p.exists():
            ensure_ascii_or_utf8(p, content)

    # Keep project brief from user prompt if missing
    if not PROJECT_BRIEF_MD.exists():
        ensure_ascii_or_utf8(
            PROJECT_BRIEF_MD,
            "# Layer 0\nTODO\n\n# Layer 1\nTODO\n\n# Layer 2\nTODO\n",
        )

    if not PROJECT_BRIEF_YAML.exists():
        ensure_ascii_or_utf8(
            PROJECT_BRIEF_YAML,
            json.dumps({"project_type": "scratch_like_game"}, indent=2),
        )

    # Seed minimal prompts/skills for Design B guardrails.
    role_keys = [
        "release_engineer",
        "planner",
        "requirements_analyst",
        "ux_designer",
        "frontend_dev",
        "backend_dev",
        "qa_tester",
        "docs_writer",
        "prompt_tuner",
    ]
    for rk in role_keys:
        d = PROMPTS_ROOT / rk
        d.mkdir(parents=True, exist_ok=True)
        v1 = d / "v1.txt"
        v2 = d / "v2.txt"
        if not v1.exists():
            v1.write_text(f"Role {rk}: produce deterministic changes only within allowlist.\n", encoding="utf-8")
        if not v2.exists():
            v2.write_text(f"Role {rk}: minimal edit strategy, no forbidden path touches.\n", encoding="utf-8")

        sd = SKILLS_ROOT / rk
        sd.mkdir(parents=True, exist_ok=True)
        sk = sd / "SKILL.md"
        if not sk.exists():
            sk.write_text(
                "---\n"
                f"name: {rk}\n"
                "description: Deterministic role execution constrained by allowlists and validators.\n"
                "---\n"
                "Follow project brief and never bypass gating.\n",
                encoding="utf-8",
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic Codex orchestrator")
    parser.add_argument("--mode", choices=["A", "B"], default="B")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--schema-mode", action="store_true")
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args(argv)

    verbose = bool(args.verbose and not args.quiet)
    design_b = args.mode == "B"

    ORCH_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    EVALS_ROOT.mkdir(parents=True, exist_ok=True)

    if not args.skip_bootstrap:
        bootstrap_required_artifacts()

    policy = load_json_file(POLICY_PATH, default_policy())

    features, feature_err = detect_codex_support()
    if feature_err and verbose:
        print(f"[warn] {feature_err}; falling back to minimal codex command")

    brief_cfg, brief_cfg_err = parse_brief_config()
    if brief_cfg_err:
        print(BRIEF_YAML_INVALID)
        return 2

    backend_required = True
    if isinstance(brief_cfg, dict) and isinstance(brief_cfg.get("backend_required"), bool):
        backend_required = bool(brief_cfg["backend_required"])
    else:
        # deterministic inference from brief text
        backend_required = "Backend REQUIRED" in read_text(PROJECT_BRIEF_MD)

    steps = make_steps(backend_required=backend_required, design_b=design_b)

    prompt_map = {"run_id": args.run_id, "steps": {}}
    total_retries = 0
    total_fixer_runs = 0
    total_changed_files = 0

    # Prompt library bootstrap in Design B if missing/empty
    if design_b:
        prompts_missing = (not PROMPTS_ROOT.exists()) or (not any(PROMPTS_ROOT.rglob("*.txt")))
        skills_missing = (not SKILLS_ROOT.exists()) or (not any(SKILLS_ROOT.rglob("SKILL.md")))
        if prompts_missing or skills_missing:
            bootstrap_required_artifacts()

    for step in steps:
        if step.optional and step.name == "backend" and not backend_required:
            if verbose:
                print("[skip] backend step not required by brief")
            continue
        if step.optional and step.name == "prompt_tuner":
            # tuner is handled by explicit Design B loop after baseline
            continue

        res = run_single_step(
            step=step,
            run_id=args.run_id,
            policy=policy,
            features=features,
            backend_required=backend_required,
            design_b=design_b,
            schema_mode=args.schema_mode,
            brief_cfg=brief_cfg,
            prompt_map=prompt_map,
            verbose=verbose,
        )

        total_retries += res.retries_used
        total_fixer_runs += res.fixer_runs
        total_changed_files += len(res.changed_paths)

        if not res.success:
            write_prompt_map(args.run_id, prompt_map)
            POLICY_PATH.write_text(json.dumps(policy, indent=2), encoding="utf-8")
            if verbose:
                print(f"[fail] step={step.name} errors={res.error_codes}")
            return 1

    # Design B eval-gated prompt tuner loop
    if design_b:
        baseline_valid = len(validate_required_outputs(design_b=True)) == 0
        cmds, cmd_err = extract_test_commands(brief_cfg)
        tests_ok = False
        if cmd_err is None:
            tests_ok, _ = run_test_commands(cmds, verbose=verbose)

        baseline_score = compute_design_b_score(
            all_valid=baseline_valid,
            tests_ok=tests_ok,
            retries=total_retries,
            fixer_runs=total_fixer_runs,
            changed_files_total=total_changed_files,
            hard_invalid=False,
        )

        baseline_eval = {
            "run_id": args.run_id,
            "phase": "baseline",
            "score": baseline_score,
            "timestamp": now_iso(),
        }
        (EVALS_ROOT / f"{args.run_id}.json").write_text(json.dumps(baseline_eval, indent=2), encoding="utf-8")

        tuner_step = Step(
            name="prompt_tuner",
            role="Prompt Tuner",
            allowlist=["prompts/**", ".codex/skills/**"],
            required_validators=["prompt_guardrails"],
            optional=True,
        )

        tuner_res = run_single_step(
            step=tuner_step,
            run_id=args.run_id,
            policy=policy,
            features=features,
            backend_required=backend_required,
            design_b=design_b,
            schema_mode=args.schema_mode,
            brief_cfg=brief_cfg,
            prompt_map=prompt_map,
            verbose=verbose,
        )

        if tuner_res.success:
            tuned_valid = len(validate_required_outputs(design_b=True)) == 0
            tests_ok2 = False
            if cmd_err is None:
                tests_ok2, _ = run_test_commands(cmds, verbose=verbose)
            tuned_score = compute_design_b_score(
                all_valid=tuned_valid,
                tests_ok=tests_ok2,
                retries=total_retries + tuner_res.retries_used,
                fixer_runs=total_fixer_runs + tuner_res.fixer_runs,
                changed_files_total=total_changed_files + len(tuner_res.changed_paths),
                hard_invalid=False,
            )
            if tuned_score <= baseline_score:
                # deterministic revert of tuner edits
                git_cmd(["restore", "--staged", "--worktree", "--", "prompts", ".codex/skills"])
            else:
                (EVALS_ROOT / f"{args.run_id}.json").write_text(
                    json.dumps(
                        {
                            "run_id": args.run_id,
                            "phase": "tuned",
                            "baseline_score": baseline_score,
                            "tuned_score": tuned_score,
                            "accepted": True,
                            "timestamp": now_iso(),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

    write_prompt_map(args.run_id, prompt_map)
    POLICY_PATH.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    if verbose:
        print("[ok] pipeline completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

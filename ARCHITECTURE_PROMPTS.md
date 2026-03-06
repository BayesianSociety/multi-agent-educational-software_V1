# Architecture Prompts

## Prompt variant loading
- Prompt variants are loaded in `orchestrator.py` via:
  - `load_variants_for_role()`
  - source precedence:
    1. Design B: `/prompts/<agent>/**` (if present)
    2. Design A external templates: `/.orchestrator/prompt_templates/<agent>/**`
    3. Internal embedded variants from `internal_variants()`

## Prompt variant selection
- Selection entrypoint: `select_variant()`.
- It implements deterministic two-phase logic:
  - BOOTSTRAP round-robin (`bootstrap_min_trials_per_variant`)
  - Post-bootstrap strategy from `policy.json`:
    - `pick_ucb1()`
    - `pick_explore_then_commit()`
    - `pick_rr_elimination()`
- Variant stats are stored in `/.orchestrator/policy.json`, keyed by role and `prompt_epoch_id`.

## Prompt assembly
- Specialist prompts are assembled in `build_step_prompt()`.
- The assembled prompt includes:
  - role-specific selected variant text
  - resolved skill instructions from `/.codex/skills/<agent>/SKILL.md` via `resolve_skill()`
  - project brief excerpt from `read_project_brief_excerpt()`
  - deterministic manifest excerpt from `hash_manifest_excerpt()`
- Final per-step prompt mapping is written to:
  - `/.orchestrator/runs/<run_id>/prompt_map.json`
  - includes `variant_id`, `variant_source`, `prompt_epoch_id`, `skill_path`, `skill_hash`, `skill_used`, `skill_excerpt_mode`.

# Architecture Prompts

- Sub-prompts are loaded in `orchestrator.py` via `load_variants(...)`.
- Variant selection is implemented in `select_variant(...)` with deterministic strategies.
- Step prompt assembly is implemented in `build_step_prompt(...)`.
- Variant source resolution order:
  1. Design B: `/prompts/<agent>/*.txt` (if present)
  2. `/.orchestrator/prompt_templates/<agent>/*.txt`
  3. Embedded templates in `embedded_variants(...)`
- Skill injection path is `/.codex/skills/<agent>/SKILL.md` via `load_skill(...)`.

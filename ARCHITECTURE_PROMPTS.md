# Prompt Architecture
- Sub-prompts are loaded per role from `/prompts/<agent>/*.txt` in Design B, otherwise from internal templates or `/.orchestrator/prompt_templates/<agent>/*.txt` in Design A.
- Variant selection is handled by deterministic policy functions in `orchestrator.py` (`choose_variant`, `select_ucb1`, `select_explore_then_commit`, `select_rr_elimination`).
- Final step prompt assembly is performed in `build_prompt`, which injects project brief content and a hashed manifest.

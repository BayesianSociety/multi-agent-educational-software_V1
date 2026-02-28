# Global Rules
- Do not modify /.orchestrator/**
- Do not modify .git/**
- Follow step allowlists exactly.
- Never mark a step complete when validators or tests fail.

# File Boundaries
- Specialists may edit only the files/paths allowlisted by the orchestrator step.
- Prompt Library Bootstrap and Prompt Tuner may edit only `/prompts/**` and `/.codex/skills/**`.
- After bootstrap, `AGENTS.md`, `PROJECT_BRIEF.md`, and `PROJECT_BRIEF.yaml` are locked.

# How to Run Tests
- Read `TEST.md` and run deterministic commands in order.
- Do not add undocumented test commands.

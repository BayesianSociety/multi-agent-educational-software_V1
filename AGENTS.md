# Global Rules
- Follow deterministic validators and step allowlists.
- Do not mark work complete if tests/validators fail.
- Do not modify /.orchestrator/**.

# File Boundaries
- Specialists may only modify paths allowlisted for their step.
- Only prompt library bootstrap/tuner may modify /prompts/** and /.codex/skills/**.
- AGENTS.md is locked after bootstrap.

# How to Run Tests
- Use commands documented in TEST.md under "# How to run tests".

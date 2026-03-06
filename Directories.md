# Exact File And Directory Creation Map With Ownership

Repository root path:
`/home/postnl/multi-agent-educational-software_V1`

## Create these yourself before running the full Design B pipeline

### Prompt 4 input files (already in your repository)
- `/home/postnl/multi-agent-educational-software_V1/Prompt_4_separate_pipeline_engine_from_project_pack.txt`
- `/home/postnl/multi-agent-educational-software_V1/Prompt_4_project_brief.txt`

### Bootstrapping files you should provide now
- `/home/postnl/multi-agent-educational-software_V1/PROJECT_BRIEF.md`
- `/home/postnl/multi-agent-educational-software_V1/PROJECT_BRIEF.yaml`

### Design B prompt library files you should create now
- `/home/postnl/multi-agent-educational-software_V1/prompts/release_engineer/variant_01.txt`
- `/home/postnl/multi-agent-educational-software_V1/prompts/release_engineer/variant_02.txt`
- `/home/postnl/multi-agent-educational-software_V1/prompts/planner/variant_01.txt`
- `/home/postnl/multi-agent-educational-software_V1/prompts/planner/variant_02.txt`
- `/home/postnl/multi-agent-educational-software_V1/prompts/requirements_analyst/variant_01.txt`
- `/home/postnl/multi-agent-educational-software_V1/prompts/requirements_analyst/variant_02.txt`
- `/home/postnl/multi-agent-educational-software_V1/prompts/ux_designer/variant_01.txt`
- `/home/postnl/multi-agent-educational-software_V1/prompts/ux_designer/variant_02.txt`
- `/home/postnl/multi-agent-educational-software_V1/prompts/frontend_dev/variant_01.txt`
- `/home/postnl/multi-agent-educational-software_V1/prompts/frontend_dev/variant_02.txt`
- `/home/postnl/multi-agent-educational-software_V1/prompts/backend_dev/variant_01.txt`
- `/home/postnl/multi-agent-educational-software_V1/prompts/backend_dev/variant_02.txt`
- `/home/postnl/multi-agent-educational-software_V1/prompts/qa_tester/variant_01.txt`
- `/home/postnl/multi-agent-educational-software_V1/prompts/qa_tester/variant_02.txt`
- `/home/postnl/multi-agent-educational-software_V1/prompts/docs_writer/variant_01.txt`
- `/home/postnl/multi-agent-educational-software_V1/prompts/docs_writer/variant_02.txt`
- `/home/postnl/multi-agent-educational-software_V1/prompts/prompt_library_bootstrap/variant_01.txt`
- `/home/postnl/multi-agent-educational-software_V1/prompts/prompt_library_bootstrap/variant_02.txt`

### Design B skill files you should create now
- `/home/postnl/multi-agent-educational-software_V1/.codex/skills/release_engineer/SKILL.md`
- `/home/postnl/multi-agent-educational-software_V1/.codex/skills/planner/SKILL.md`
- `/home/postnl/multi-agent-educational-software_V1/.codex/skills/requirements_analyst/SKILL.md`
- `/home/postnl/multi-agent-educational-software_V1/.codex/skills/ux_designer/SKILL.md`
- `/home/postnl/multi-agent-educational-software_V1/.codex/skills/frontend_dev/SKILL.md`
- `/home/postnl/multi-agent-educational-software_V1/.codex/skills/backend_dev/SKILL.md`
- `/home/postnl/multi-agent-educational-software_V1/.codex/skills/qa_tester/SKILL.md`
- `/home/postnl/multi-agent-educational-software_V1/.codex/skills/docs_writer/SKILL.md`
- `/home/postnl/multi-agent-educational-software_V1/.codex/skills/prompt_library_bootstrap/SKILL.md`

## Created by the orchestrator run (Codex specialists under orchestrator control)

### Root files
- `/home/postnl/multi-agent-educational-software_V1/REQUIREMENTS.md`
- `/home/postnl/multi-agent-educational-software_V1/TEST.md`
- `/home/postnl/multi-agent-educational-software_V1/AGENT_TASKS.md`
- `/home/postnl/multi-agent-educational-software_V1/README.md`
- `/home/postnl/multi-agent-educational-software_V1/RUNBOOK.md`
- `/home/postnl/multi-agent-educational-software_V1/.pipeline_plan.json`
- `/home/postnl/multi-agent-educational-software_V1/.pipeline_plan_schema.json`
- `/home/postnl/multi-agent-educational-software_V1/.env.example`
- `/home/postnl/multi-agent-educational-software_V1/.gitignore`
- `/home/postnl/multi-agent-educational-software_V1/docker-compose.yml`
- `/home/postnl/multi-agent-educational-software_V1/AGENTS.md`

### Application directories and implementation
- `/home/postnl/multi-agent-educational-software_V1/design`
- `/home/postnl/multi-agent-educational-software_V1/frontend`
- `/home/postnl/multi-agent-educational-software_V1/backend`
- `/home/postnl/multi-agent-educational-software_V1/tests`
- `/home/postnl/multi-agent-educational-software_V1/backend/prisma/schema.prisma`
- `/home/postnl/multi-agent-educational-software_V1/backend/prisma/migrations`

## Created by the orchestrator process itself (private state)
- `/home/postnl/multi-agent-educational-software_V1/.orchestrator/policy.json`
- `/home/postnl/multi-agent-educational-software_V1/.orchestrator/runs/<run_id>/...`
- `/home/postnl/multi-agent-educational-software_V1/.orchestrator/evals/<run_id>.json`

## Note
The Prompt 4 Design B spec allows prompt and skill libraries to be bootstrap-created by the orchestrator. In your current implementation, pre-creating them yourself is the safest path before running `python3 orchestrator.py --design-b --mode verbose`.

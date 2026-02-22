# multi-agent-educational-software_V1
cat Prompt_4_separate_pipeline_engine_from_project_pack.txt Prompt_4_project_brief.txt \
  | codex exec --full-auto --json -o ./last_message.txt 

# potential flags in the prompt files  
• Prompt_4_project_brief.txt does not define concrete CLI run flags. It defines requirements that generated docs
  must include startup commands for frontend/backend/tests/docker (Prompt_4_project_brief.txt:55).

  Prompt_4_separate_pipeline_engine_from_project_pack.txt defines these run-argument types:

  1. codex exec flags and invocation args (Prompt_4_separate_pipeline_engine_from_project_pack.txt:46,
     Prompt_4_separate_pipeline_engine_from_project_pack.txt:55,
     Prompt_4_separate_pipeline_engine_from_project_pack.txt:227)

  - --sandbox workspace-write
  - - (stdin prompt input argument, not a flag)
  - optional --experimental-json
  - optional --output-schema
  - optional --ask-for-approval
  - optional --config
  - --help (for feature detection)

  2. Orchestrator runtime visibility mode flags (Prompt_4_separate_pipeline_engine_from_project_pack.txt:243)

  - --quiet
  - --verbose

  3. Git subprocess args used for integrity checks (Prompt_4_separate_pipeline_engine_from_project_pack.txt:151,
     Prompt_4_separate_pipeline_engine_from_project_pack.txt:163)

  - --cached
  - --name-only

  4. Non-CLI run configuration arguments (config keys/enums that affect run behavior)

  - selection_strategy: ucb1 | explore_then_commit | rr_elimination
    (Prompt_4_separate_pipeline_engine_from_project_pack.txt:445)
  - tunables: bootstrap_min_trials_per_variant, ucb_c, commit_window_runs, elim_min_trials, elim_min_mean_clean,
    elim_max_failure_rate (Prompt_4_separate_pipeline_engine_from_project_pack.txt:435,
    Prompt_4_separate_pipeline_engine_from_project_pack.txt:449,
    Prompt_4_separate_pipeline_engine_from_project_pack.txt:457,
    Prompt_4_separate_pipeline_engine_from_project_pack.txt:466)
  - PROJECT_BRIEF.yaml knobs: project_type (examples: scratch_like_game, financial_app, research_tool) and tests
    profile mode {"tests":{"command_source":"profile","commands":[...]}}
    (Prompt_4_separate_pipeline_engine_from_project_pack.txt:97,
    Prompt_4_separate_pipeline_engine_from_project_pack.txt:375)

  That is the full set explicitly present in the two files.
  
  
# quality - this is a very good point for Framework_for_quality_evaluation
  What would raise product-quality confidence:

  1. Add behavior-focused validators

  - Validate key user flows end-to-end (dog reaches bone, failure states, reset behavior).
  - Validate deterministic execution traces, not just pass/fail.

  2. Strengthen test quality gates

  - Minimum coverage thresholds.
  - Required test categories (unit + integration + UI flow).
  - Mutation/property tests for core execution logic.

  3. Add code quality gates

  - Typecheck, lint, static analysis, dependency audit.
  - Performance/accessibility checks for frontend.

  4. Add architectural checks

  - Enforce API contracts.
  - Check frontend actually calls backend endpoints.
  - Validate error handling and edge cases.

  5. Add review criteria beyond files/headings

  - Explicit rubrics for maintainability, readability, and security.

  Net: your prompts are a strong “factory line,” but to consistently get truly solid software, you need stricter
  “quality inspection” criteria inside that line.
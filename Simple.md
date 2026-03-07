I read the whole file (`620` lines).  
Below is a clear table of contents with purpose, why it exists, and whether to keep it.

**Table Of Contents Analysis**

| Section in file | What it does | Why it is needed | Is it really needed |
|---|---|---|---|
| `CONTEXT / SPECIFICATION` preamble | Tells the model to read first and execute after reading | Prevents partial reading and early action | Keep, but shorten to one direct instruction |
| `Version 3.1` and design split intro | Explains two modes: core mode and extension mode | Defines scope and prevents mode confusion | Keep |
| `What this gives you` | Describes outcomes of core and extension modes | Sets expectations before details | Keep, but simplify wording |
| `PROJECT LAYERS` | Defines Layer 0 to Layer 3 structure | Separates stable constraints from evolving content | Keep, this is very valuable |
| `Non-game interpretation note` | Maps game words to general product words | Reuse for non game projects | Keep if you want one universal engine prompt |
| `Layer placement rule of thumb` | States which layers must be fixed early | Reduces drift in core constraints | Keep |
| `Prerequisites` | Declares required tools and environment | Prevents impossible runs | Keep |
| `Compatibility note` | Handles optional command line flags and fallback behavior | Makes engine resilient across command line versions | Keep |
| `Goal` | Defines the main outcome: one orchestrator script | Prevents scope creep | Keep |
| `Hard constraints` | Bans software development kit calls and natural language gating | Preserves deterministic behavior | Keep |
| `Required outputs in core mode` | Lists mandatory product files and directories | Gives concrete completion targets | Keep |
| `Additional outputs in extension mode` | Adds prompt and skill library artifacts | Needed only for self improving extension mode | Keep, but isolate in extension prompt |
| `Project brief artifact` | Introduces locked `PROJECT_BRIEF.md` | Central source of project truth | Keep, very important |
| `Project brief config artifact` | Adds machine readable `PROJECT_BRIEF.yaml` as JavaScript Object Notation subset | Allows swapping project pack without changing engine | Keep, very important |
| `Orchestrator private state` | Defines `.orchestrator` internal files | Separates engine state from product output | Keep |
| `Non-negotiable invariants` | Defines hard fail rules and retry limits | Prevents unsafe or uncontrolled behavior | Keep |
| `Additional extension invariants` | Restricts prompt and skill edits and lock behavior | Needed only when extension mode is enabled | Keep, extension only |
| `Boundary Enforcement` run window model | Defines before and after snapshots, checks, and revert timing | This is the core safety mechanism | Keep, critical |
| `Path safety and revert of untracked files` | Blocks path traversal and symlink escape | Prevents allowlist bypass and hidden writes | Keep, critical |
| `Agent model` | Defines orchestrator role and specialist roles | Gives consistent workflow structure | Keep |
| `Default pipeline order` | Specifies step order and product acceptance gate | Avoids random sequencing | Keep |
| `Parallelization` | Allows parallel runs only for disjoint write sets | Gives speed option while preserving determinism | Keep, but can be optional section |
| `Codex invocation protocol` | Defines subprocess invocation, logging, and no natural language dependence | Ensures consistent execution behavior | Keep |
| `Project layer injection rule` | Forces brief and manifest into each step prompt | Keeps specialists aligned to project constraints | Keep |
| `Filesystem allowlists and forbidden paths` | Defines what each step may change | Prevents cross-step contamination | Keep |
| `Change limits` | Caps file count, bytes, and deletion count | Reduces blast radius of bad runs | Keep |
| `Deterministic validators` | Defines existence, structure, and timing checks | Converts quality into deterministic checks | Keep |
| `Content validators` | Enforces required headings and ownership rules | Prevents incomplete documentation and drift | Keep, but reduce strictness where not essential |
| `Product acceptance validators` | Defines behavior-level end gate and failure codes | Prevents passing with incomplete product behavior | Keep |
| `Test command extraction` | Defines exact test command parsing modes | Makes test execution deterministic | Keep |
| `Orchestrator-only learning` policy | Stores per variant and per error statistics | Enables deterministic improvement without natural language parsing | Keep |
| `Prompt variant source and selection` | Defines where variants come from and deterministic selection methods | Enables stable prompt optimization | Keep if you use variants; simplify if not |
| `Prompt map and skill injection` | Requires trace file and role skill inclusion | Improves auditability and role consistency | Keep |
| `Deterministic prompt epoching` | Resets selection state when prompt or skill content changes | Prevents stale metrics after prompt updates | Keep |
| `Fixer runs` | Defines narrow, bounded repair attempts | Useful for deterministic recovery | Keep |
| `Extension scoring and tuner loop` | Adds score function and accept or reject prompt tuning | Needed only for extension mode | Keep in extension prompt only |
| `Extension prompt and skill guardrails` | Prevents unsafe instructions inside prompts and skills | Needed for safe self improvement | Keep in extension prompt only |
| `Implementation constraints` and `must do`/`must not do` | Final coding constraints for generator | Forces implementation discipline | Keep |

**Why it feels cluttered now**

1. Core engine rules and project specific content are mixed in one document.
2. Core mode and extension mode are interleaved many times.
3. Several sections repeat the same rule in different words.
4. There is high detail in areas that should be configurable data, not fixed prose.

---

**Important parts to keep in the next version**

1. Boundary Enforcement with run windows.
2. Hard invariants and allowlists.
3. Deterministic validators and product acceptance gate.
4. Locked `PROJECT_BRIEF.md` and `PROJECT_BRIEF.yaml` model.
5. Test command extraction rules.
6. Prompt variant determinism and prompt epoching.
7. Per step logging and prompt map trace.
8. Path safety and untracked file revert rules.
9. Retry bounds and change caps.
10. Clear pipeline step order and ownership.

---

**How to split into two prompts cleanly**

1. **Engine Prompt (stable, rarely changed)**  
Keep only engine mechanics:
- command invocation behavior
- run window safety
- allowlists and invariants
- retry and rollback logic
- validator framework and timing framework
- test command extraction framework
- policy learning framework
- prompt variant selection framework
- logging artifact schema
- extension hooks only as interfaces, not project details

2. **Project Prompt (replaceable per project)**  
Keep only project contract:
- Layer 0 to Layer 3 content
- required outputs for this project
- endpoint and persistence requirements
- content and product acceptance checks specific to this project
- role specific prompt and skill content targets

3. **Contract between both prompts**  
Use only these shared interfaces:
- `PROJECT_BRIEF.md` for human readable constraints
- `PROJECT_BRIEF.yaml` for machine readable toggles
- fixed error code naming style
- fixed run artifact paths under `.orchestrator`

This gives you one stable engine prompt and many swappable project prompts without rewriting engine logic each time.

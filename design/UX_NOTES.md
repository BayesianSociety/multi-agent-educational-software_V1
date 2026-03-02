# Scratch-like Editor UX Notes

## Layout
- Left panel: block palette with exactly two draggable block types for MVP (`Move(1)`, `Jump`).
- Center panel: single vertical program lane with top-to-bottom execution order.
- Right panel: play area showing dog, bone, obstacles, and `background.png` alignment.
- Bottom/toolbar controls: `Run` and `Reset` buttons always visible.

## Visual Hierarchy (Ages 7-12)
- Use large, readable labels for blocks and controls (minimum 16px body equivalent).
- Pair icon + text for key actions (`Run`, `Reset`, `Move`, `Jump`) to reduce reading load.
- Keep one sentence of narrative text visible per level above the play area.
- Do not rely on color alone for state changes; include border/icon/text indicators.

## Core Interaction Flow
1. Player drags or inserts `Move(1)` and `Jump` blocks into the center lane.
2. Player starts execution with `Run`.
3. Runtime highlights the current block being interpreted.
4. Dog animation advances in deterministic, step-by-step order.
5. On failure, message explains what happened and user edits sequence.
6. On success, level-complete state is shown and progress is persisted.

## Deterministic Execution Feedback
- Highlight only one current block at a time.
- Advance highlight strictly in sequence index order.
- Show execution pointer text (example: `Step 2 of 4`) near the workspace.
- Freeze editing controls while a run is active; re-enable on completion/failure.
- Reset clears runtime state and returns highlight to pre-run idle state.

## Accessibility Baseline
- All core actions are keyboard operable:
  - Tab/Shift+Tab moves focus through palette, workspace, and controls.
  - Enter/Space activates focused controls.
  - Keyboard-add path for blocks must exist (not drag-only).
- Provide visible focus ring with sufficient contrast on interactive elements.
- Narration/help text is always present as on-screen text (not audio-only).
- Error and success feedback is textual and persistent long enough to read.

## Tutorial UX
- Level 1 introduces only `Move(1)` with one short instruction sentence.
- Level 2 introduces `Jump` with one short instruction sentence.
- First-time hint text should be dismissible and non-blocking.
- Hint language should be concrete and action-based (example: "Add 2 Move blocks").

## Failure and Retry States
- Failure panel includes:
  - What failed (example: hit obstacle or did not reach bone).
  - Where it failed (step number).
  - Prompt to edit sequence and retry.
- `Reset` returns dog position and run-state immediately with no randomness.

## Responsive Behavior
- Desktop: three-column layout (palette, workspace, play area).
- Small screens/tablets: stack panels in this order: narrative + controls, workspace, play area, palette.
- Ensure `Run` and `Reset` remain reachable without precision dragging.

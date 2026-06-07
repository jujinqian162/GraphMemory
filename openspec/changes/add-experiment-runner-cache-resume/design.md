## Context

The user-facing entry point is `scripts/experiment.py`. It loads or initializes a run manifest, expands a selected workflow into `StageCommand` objects, executes those commands, and updates `manifest.json` status metadata afterward.

Status inspection already reports `missing`, `complete`, `stale`, and `alias` rows, and retrieval status checks its run summary. Other stages still rely mostly on output-file existence, even though the low-level scripts write run summaries with inputs, outputs, effective config, script name, and success status. This change makes those summaries part of the runner's resume evidence.

## Goals / Non-Goals

**Goals:**
- Make `run` cache-aware by default: repeated full-stage invocations resume from the first incomplete or stale planned command.
- Add `--no-cache` to `run` and `plan`, and make the help text explicit enough for discovery from `--help`.
- Validate stage completion with live artifact evidence and run summaries where those summaries exist.
- Keep types narrow: use typed dataclasses and closed enums for command/status relationships instead of unstructured `dict[str, Any]` at the new abstraction boundary.

**Non-Goals:**
- Do not add content hashing, artifact digests, or a generic DAG cache engine.
- Do not make low-level scripts read manifests or decide whether to skip work.
- Do not change retriever, training, pair-building, or evaluation runtime semantics.
- Do not silently backfill missing upstream stages when the user explicitly selects a downstream range.

## Decisions

### Decision 1: Add a small resume/cache helper after planning

`build_stage_plan(...)` continues to describe the complete selected lifecycle. A new helper receives the manifest and the concrete command sequence, inspects live status, and returns a typed resume decision with:

- selected commands to execute,
- skipped completed-prefix commands,
- the first non-complete command, if any.

This keeps workflow planning explicit and avoids introducing a planner-to-status import cycle. The CLI can use the same helper for `plan` rendering and `run` execution.

Alternative considered: put cache filtering inside `planner.py`. Rejected because `status.py` already imports planner helpers for variant status; adding the reverse dependency would either create a cycle or force a broader module split.

### Decision 2: Prune only the completed prefix

The default cache mode skips only commands before the first command whose live status is not complete or alias. From that command onward, the runner executes the rest of the selected plan.

This is intentionally more conservative than independently skipping every complete command. If an upstream command reruns and overwrites an artifact at the same path, downstream summaries may still point at the same paths; prefix pruning avoids reusing potentially stale downstream artifacts after a rerun boundary.

Alternative considered: per-command skip with dependency propagation. Rejected for this change because it requires a split/method/variant dependency graph and adds complexity around ablation aliases.

### Decision 3: Use explicit typed status keys

Add a small immutable status key type shaped like `stage + optional split + optional method + optional variant`. This replaces stringly status joins at the new boundary while preserving the existing manifest `stage_status` JSON shape.

The key is computed from both `StageCommand` and status rows. The helper treats `complete` and `alias` as reusable, and treats `missing` and `stale` as rerun boundaries.

### Decision 4: Promote run-summary checks incrementally but consistently

Each status row should use stage-specific run-summary checks when the corresponding low-level script writes a summary:

- `prepare`: raw input, generated input/labels/combined paths, max examples, seed, and offset.
- `graphs`: task input, graph output, graph config fields.
- `pairs`: train tasks/labels/graphs, pairs/summary outputs, and effective pair sampling config.
- `train`: train/dev inputs, train pairs, metrics/checkpoint outputs, and effective training config.
- `tune`: dev inputs, selected config output, method, top-k, dense encoder settings, and search-space path.
- `retrieve`: keep and reuse the existing prediction summary validation.
- `evaluate`: prediction/labels/graphs inputs, metrics/debug outputs, and failure-case limit.
- `aggregate`: metrics input directory and table outputs.

When a legacy output exists but its run summary is missing, the status can remain `complete` only for stages where there is no better evidence path yet. For runner-created artifacts covered by this change, missing or mismatched summaries should be reported as `stale`.

## Risks / Trade-offs

- [Risk] Existing historical artifacts without summaries may become stale under stricter checks. -> Mitigation: scope strict summary requirements to paths with known runner summaries and keep errors actionable in status output.
- [Risk] Prefix-only pruning may rerun more commands than strictly necessary. -> Mitigation: this is safer for provenance and still solves the costly common case where a long run was interrupted before later stages.
- [Risk] New CLI behavior could surprise users who expect repeated `run` to rerun everything. -> Mitigation: expose `--no-cache` on `run` and `plan`, document it, and make help text explicit.
- [Risk] Help output or parser contracts may drift. -> Mitigation: freeze parser options and verify `--help` output in tests and smoke checks.

## Migration Plan

1. Add failing tests for parser discoverability, default prefix resume, no-cache behavior, stale summary boundaries, and stricter status checks.
2. Implement typed status keys and cache-resume helper.
3. Strengthen status validators for stage run summaries.
4. Wire `--no-cache` into `plan` and `run`.
5. Update docs and OpenSpec tasks.
6. Run focused tests, full validation, and a smoke-profile workflow that proves initial run, cached second run, and no-cache plan behavior.

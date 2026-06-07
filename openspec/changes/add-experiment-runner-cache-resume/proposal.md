## Why

Named experiment runs already write manifest-backed artifact paths and per-stage run summaries, but `scripts/experiment.py run` still executes every planned command by default. Long or interrupted runs should resume from the first incomplete or stale planned command without forcing users to hand-select `--from` ranges.

## What Changes

- Add default cache-aware resume for `scripts/experiment.py run`: the runner inspects live artifact status and skips the completed prefix of the selected command plan.
- Add a discoverable `--no-cache` flag to `run` and `plan` so users can request the full unpruned plan or force all selected commands to execute.
- Strengthen status/provenance checks so completed stages are validated with run summaries when available, not only by output-file existence.
- Keep low-level scripts and their explicit input/output contracts unchanged.

## Capabilities

### New Capabilities
- `experiment-runner-cache-resume`: Default cache-aware resume and explicit no-cache controls for named experiment runs.

### Modified Capabilities
- None.

## Impact

- Affected code: `scripts/experiment.py`, `scripts/workflow/status.py`, a small workflow resume/cache helper, workflow exports, tests, and command documentation.
- User-facing API: new `--no-cache` option on `scripts/experiment.py run` and `scripts/experiment.py plan`.
- No new runtime dependencies and no changes to retriever, training, or low-level script artifact contracts.

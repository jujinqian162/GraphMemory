## Why

The current Phase 1 pipeline is contract-safe but operationally painful: users must hand-write long command sequences with repeated `--input` and `--output` paths, and exploratory runs can accidentally reuse artifacts from another run.

This change introduces a high-level experiment runner that keeps low-level artifact contracts explicit while giving users a named, isolated run directory with stable generated paths, resumable stages, and clearer config ownership.

## What Changes

- Add a user-facing experiment runner script as the normal way to run evidence-retrieval experiments.
- Use named run directories under `runs/` so intermediate artifacts, tuned configs, predictions, metrics, debug files, and summaries are isolated per experiment.
- Generate a per-run manifest that records effective config, fixed artifact paths, selected stages, selected methods, and provenance checks.
- Support stage-based execution so users can run full workflows, resume from a stage, or combine selected stages and methods without manually constructing every low-level command.
- Keep existing low-level scripts and their explicit input/output arguments as the contract boundary.
- Reorganize config roles so stable experiment defaults, tuning search spaces, and published selected configs have distinct locations and names.
- Update command and reproducibility documentation to make the experiment runner the recommended path while preserving low-level command examples for debugging and contract review.

## Capabilities

### New Capabilities

- `experiment-runner`: Provides named experiment runs, isolated artifact layout, manifest-based provenance, config layering, stage orchestration, status inspection, and dry-run command planning.

### Modified Capabilities

- None.

## Impact

- Affected code:
  - `scripts/experiment.py`
  - new or updated orchestration helpers under `graph_memory/`
  - `graph_memory/io.py` if config loading/merging needs small shared helpers
  - tests for runner planning, manifest generation, path isolation, and resume behavior
- Affected configs:
  - replace ambiguous Phase 1 config names with clearer experiment defaults, tuning search spaces, and published selected configs
  - move run-specific tuned configs into `runs/<experiment>/tuned/`
- Affected docs:
  - `docs/40-operations/commands.md`
  - `docs/40-operations/reproducibility.md`
  - `docs/40-operations/implementation-handoff.md`
  - root `README.md`
- No breaking changes to existing low-level CLI arguments, JSON artifact schemas, ranked-result schemas, evaluation metrics, or method names.

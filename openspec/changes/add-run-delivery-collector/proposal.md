## Why

Full train runs can contain large intermediate artifacts that are impractical to copy from the HPC server, while the report analysis only needs compact audit artifacts, metrics, summaries, and capped debug records. A dedicated delivery collector makes the transfer boundary explicit and repeatable.

## What Changes

- Add a `scripts/deliver` utility that accepts `--name <train_id>`, reads `runs/<train_id>/`, and copies report-relevant run artifacts into `results/<train_id>/...` while preserving the original run-relative structure.
- Keep compact evidence needed for analysis: manifest/config, aggregate tables, per-method metric CSVs, run summaries, training config/history/summary, train-pair summaries, graph stats, selected tuning configs, and capped failure cases.
- Exclude large reproducible/intermediate artifacts by default, including raw prepared inputs, full graph JSON files, full ranked predictions, train-pair JSON, and checkpoint binaries.
- Emit a machine-readable delivery manifest listing copied files, skipped files, byte counts, and skip reasons.

## Capabilities

### New Capabilities
- `run-delivery-collector`: Selective collection of compact, report-relevant artifacts from a named experiment run for transfer and downstream analysis.

### Modified Capabilities
- None.

## Impact

- Affected code: new files under `scripts/deliver/`, focused tests under `tests/`, and OpenSpec artifacts under `openspec/changes/add-run-delivery-collector/`.
- User-facing API: a new name-based CLI for collecting `runs/<train_id>` into `results/<train_id>` by default.
- Dependencies: no new runtime dependency; the implementation uses the Python standard library.

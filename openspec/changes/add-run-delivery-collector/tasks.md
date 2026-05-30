## 1. Collector Contract

- [x] 1.1 Add tests for selecting report-relevant run artifacts while preserving run-relative paths.
- [x] 1.2 Add tests for excluding large intermediates and recording skip reasons.
- [x] 1.3 Add tests for writing an auditable delivery manifest and failing on missing run directories.

## 2. CLI Implementation

- [x] 2.1 Create `scripts/deliver/collect_run_artifacts.py` with standard-library selection, copy, and manifest logic.
- [x] 2.2 Add CLI arguments for source run directory, output root, optional report inclusion, max file size, and dry-run behavior.
- [x] 2.3 Keep default behavior scoped to compact analysis artifacts and preserve original directory structure under `results/<run_id>/`.

## 3. Verification

- [x] 3.1 Run the targeted collector tests.
- [x] 3.2 Run formatting/static checks for touched Python files.
- [x] 3.3 Smoke-run the collector on the local `runs/rgcn_full_train` tree and inspect the manifest summary.

## 4. Review Feedback

- [x] 4.1 Replace the public `--run-dir` CLI with the `--name <train_id>` convention.
- [x] 4.2 Update help/spec/design wording to document `runs/<name>` to `results/<name>` defaults.
- [x] 4.3 Re-run targeted tests, static checks, and a name-based smoke run.

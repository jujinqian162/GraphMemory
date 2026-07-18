# Verification

This record covers the corrected method-only implementation.

## Scope compatibility

- twowiki_provenance schema, converter, parser, scorer, projectors, fixture, dataset config, and dataset tests match main.
- Shared evaluation metrics, evaluation artifacts, aggregate output, and cache identity match main.
- Provenance R-GCN inference and tests match main.
- No clean-data generation, confidence audit, dataset migration, paired bootstrap, or counterfactual workflow is part of this change.

## Cache compatibility

Prefect scientific tasks use `TASK_SOURCE + INPUTS`. The following R-GCN cache-bearing surfaces are byte-identical to `main`: `experiment/tasks.py`, `experiment/workflow.py`, `stages/prepare.py`, `stages/pairs.py`, `stages/models.py`, `stages/evaluate.py`, the `twowiki_provenance` dataset config and implementation, the execution-provenance R-GCN method config, and the provenance R-GCN model/inference package. Their implementation-version inputs remain `prepare-v1`, `training-pairs-v1`, `provenance-rgcn-train-v1`, `ranking-v2-device-aware`, and `evaluation-v1`.

Therefore an already completed execution-provenance R-GCN run keeps the same prepare, pair, train, ranking, and evaluation cache keys. The only cache-bearing input changes are the resolved method configs for `graphrag` and `execution_provenance_retriever`; those methods receive new ranking keys, and their evaluation keys change only through the resulting prediction artifact reference. No dataset, pair, training, checkpoint, or R-GCN cache is invalidated.

## Focused regression

The focused method, Prefect-cache, dataset compatibility, workflow-config, and trainable provenance regression suite passed with 69 tests. The cache-refresh test also passed independently after one transient Windows temporary-directory `WinError 5` during a concurrent full-suite run.

## Final gates

- Full pytest: 131 passed, with one existing RequestsDependencyWarning.
- Ruff: passed.
- basedpyright: 0 errors, 0 warnings, 0 notes.
- compileall: passed.
- git diff --check: passed.
- Forbidden-scope diff scan: no dataset, training model, shared evaluation, dataset fixture, or dataset-test content changes.
- OpenSpec strict validation: passed.

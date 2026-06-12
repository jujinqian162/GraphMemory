## Why

The current `dense_ft` training artifact only records one final dev MAP value, so a completed run cannot show whether the model was still improving, overfitting, or simply saving the last epoch. This change adds the minimum dense-ft-owned training observability needed to interpret fit quality without expanding the workflow, delivery collector, or R-GCN training path.

## What Changes

- Record an epoch-0 dev MAP baseline before dense-ft fine-tuning begins.
- Record one dense-ft metric row per training epoch with `epoch`, `global_step`, `train_loss`, the selected dev metric, `best_epoch`, and `best_dev_metric`.
- Save `learned/dense_ft/checkpoints/best_model` according to the best dev metric instead of blindly saving the final epoch model.
- Keep the metric set intentionally minimal: no per-step logs, no full retrieval metrics, no Recall/MRR/Full Support during training.
- Keep the existing `train_metrics.jsonl` artifact path so `scripts/train_method.py` and `scripts/deliver/collect_run_artifacts.py` do not need behavior changes.
- Scope implementation to dense-ft training internals and dense-ft tests; do not change frozen dense retrieval, R-GCN training, workflow planning, registry dispatch, or delivery rules.

## Capabilities

### New Capabilities

- `dense-ft-training-observability`: Dense fine-tuning emits minimal epoch-level fit diagnostics and saves the best dev checkpoint.

### Modified Capabilities

None.

## Impact

- Affected code: `graph_memory/models/dense_finetune/training.py` and `tests/test_dense_finetune_training.py`.
- Affected artifacts: `learned/dense_ft/train_metrics.jsonl` contains multiple epoch-level rows instead of one final row; `learned/dense_ft/checkpoints/best_model` becomes dev-selected.
- Affected APIs: internal dense-ft trainer result/adapter interfaces may gain metric-record access, but public CLI and workflow commands remain unchanged.
- Dependencies: no new dependencies; continue using `sentence-transformers==2.7.0`.
- Verification: focused dense-ft training tests plus OpenSpec strict validation.

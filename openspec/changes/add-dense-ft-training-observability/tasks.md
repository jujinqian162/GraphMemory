## 1. Dense-ft Observability Tests

- [x] 1.1 Add a dense-ft training test proving `train_dense_finetune()` returns an epoch-0 baseline row plus one row per training epoch instead of a single `phase=final` row.
- [x] 1.2 Add a SentenceTransformers 2.7 adapter test proving `model.fit()` receives an epoch-end callback and does not rely on `save_best_model=True`.
- [x] 1.3 Add a fake loss-module test proving batch losses are averaged into the epoch row as `train_loss`.
- [x] 1.4 Add a best-model test proving the baseline model remains saved when epoch metrics degrade.
- [x] 1.5 Add a best-model test proving an improved epoch overwrites `checkpoints/best_model` and updates `best_epoch` / `best_dev_metric`.
- [x] 1.6 Add a comparison test proving `DenseFinetuneSelectionSettings(higher_is_better=False)` treats lower selected metric values as improvements.

## 2. Dense-ft Metric Tracking Internals

- [x] 2.1 Add a private dense-ft metric tracker in `graph_memory/models/dense_finetune/training.py` that records `epoch`, `global_step`, `train_loss`, selected metric, `best_epoch`, and `best_dev_metric`.
- [x] 2.2 Add private comparison logic that uses `DenseFinetuneSelectionSettings.higher_is_better` and initializes the best state from epoch 0.
- [x] 2.3 Add a private loss observer for `MultipleNegativesRankingLoss` that accumulates scalar loss weighted by batch size and resets after each epoch row.
- [x] 2.4 Ensure the loss observer removes hooks or wrappers after `fit()` finishes, including failure paths.

## 3. SentenceTransformers 2.7 Adapter Update

- [x] 3.1 Update `_SentenceTransformers27FitRunner` to evaluate the epoch-0 dev metric before calling `model.fit()`.
- [x] 3.2 Save the epoch-0 model to `model_dir` before optimizer steps so the base model is a valid best checkpoint candidate.
- [x] 3.3 Pass an epoch-end callback to `model.fit()` that normalizes SentenceTransformers zero-based epochs to one-based metric rows.
- [x] 3.4 Compute cumulative `global_step` from the completed epoch and train loader length.
- [x] 3.5 Manually save `model_dir` from the callback only when the selected dev metric improves over the current best.
- [x] 3.6 Keep `SentenceTransformer.fit()` as the only training loop; do not copy optimizer, scheduler, AMP, or batching logic from SentenceTransformers.
- [x] 3.7 Remove the final-only `trainer.evaluate()` metric synthesis path from `train_dense_finetune()` and return the adapter-owned metric rows.

## 4. Boundary Preservation

- [x] 4.1 Verify `scripts/train_method.py` still writes `DenseFinetuneTrainingResult.metric_records` without dense-ft-specific branching.
- [x] 4.2 Verify `scripts/deliver/collect_run_artifacts.py` already copies `learned/*/train_metrics.jsonl` and requires no change.
- [x] 4.3 Verify workflow planning still points dense-ft training metrics at `learned/dense_ft/train_metrics.jsonl`.
- [x] 4.4 Do not modify R-GCN training code unless a failing dense-ft-focused test proves an unavoidable shared-contract issue.

## 5. Verification

- [x] 5.1 Run `uv run pytest tests/test_dense_finetune_training.py -q`.
- [x] 5.2 Run `uv run pytest tests/test_deliver_run_artifacts.py tests/test_dense_ft_workflow.py -q` to confirm artifact and workflow boundaries.
- [x] 5.3 Run `uv run basedpyright graph_memory scripts tests --level error`.
- [x] 5.4 Run `openspec validate add-dense-ft-training-observability --strict`.
- [x] 5.5 Run `git diff --check`.

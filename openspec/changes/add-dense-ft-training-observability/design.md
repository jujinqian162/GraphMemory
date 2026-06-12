## Context

`dense_ft` training currently delegates optimization to `SentenceTransformer.fit()` and then runs one final `InformationRetrievalEvaluator` call. The resulting `train_metrics.jsonl` contains only one `phase=final` row, so a delivered run cannot show whether dev quality improved across epochs or whether the saved model came from the best epoch.

The existing workflow already has the right artifact path: `learned/dense_ft/train_metrics.jsonl` is written by `scripts/train_method.py` and copied by `scripts/deliver/collect_run_artifacts.py`. The missing behavior lives inside the dense-ft trainer adapter, so this change should not alter the train script, workflow manifest, artifact collector, registry, or R-GCN training path.

## Goals / Non-Goals

**Goals:**

- Emit minimal dense-ft epoch-level fit diagnostics in `train_metrics.jsonl`.
- Include an epoch-0 dev MAP baseline before any dense-ft optimizer step.
- Record one row per training epoch with train loss, dev selected metric, global step, best epoch, and best dev metric.
- Make `checkpoints/best_model` mean the model with the best dense-ft dev metric, including epoch 0.
- Keep the implementation local to `graph_memory.models.dense_finetune.training` and dense-ft tests.

**Non-Goals:**

- Do not add step-level logs, TensorBoard, CSV evaluator output, or full retrieval metrics during training.
- Do not compute `Recall@5/10`, `MRR`, or `Full Support` per epoch.
- Do not change dense-ft config schema or expose new CLI flags.
- Do not modify `scripts/train_method.py`, workflow planning, delivery collection, or R-GCN training.
- Do not replace `SentenceTransformer.fit()` with a copied training loop.

## Decisions

### Decision: Dense-ft trainer owns metric records

`DenseFinetuneTrainer` will expose epoch metric records after `train()`. `train_dense_finetune()` will read those records instead of synthesizing a single final row. This keeps `scripts/train_method.py` unchanged because it already serializes `DenseFinetuneTrainingResult.metric_records`.

Alternative considered: make `scripts/train_method.py` query dense-ft-specific state. Rejected because the train script should remain method-agnostic and already has a generic result contract.

### Decision: Record epoch 0 before calling `fit()`

The SentenceTransformers evaluator will run once before optimizer steps and produce the epoch-0 selected metric row. The current model will be saved to `model_dir` as the initial best candidate before training starts.

Alternative considered: use only SentenceTransformers epoch-end callback. Rejected because it cannot distinguish "training improved over the base model" from "first epoch is the only observed model."

### Decision: Use callback for epoch-end dev metric and manual best-model save

`SentenceTransformer.fit()` will receive a callback that records epoch-end dev scores. The adapter will pass `output_path` only for evaluator output isolation if needed, but it will not rely on `save_best_model=True`; instead, the callback compares against the tracker initialized by epoch 0 and calls `model.save(model_dir)` only when the selected metric improves.

Alternative considered: rely on `save_best_model=True`. Rejected because SentenceTransformers initializes `best_score` internally and has no knowledge of the epoch-0 baseline.

### Decision: Observe train loss through the loss module boundary

The adapter will wrap or hook `MultipleNegativesRankingLoss` during `fit()` to accumulate the scalar loss returned by each batch. At each epoch-end callback, it will write the average loss for that epoch and reset the accumulator.

Alternative considered: copy the `SentenceTransformer.fit()` training loop to get direct loss access. Rejected because it would duplicate optimizer, scheduler, AMP, checkpoint, and batching semantics owned by SentenceTransformers 2.7.0.

### Decision: Keep metric rows minimal and stable

Each row will use the same JSONL artifact and contain only:

- `epoch`
- `global_step`
- `train_loss`
- the configured selected metric, currently `eval_dev_cos_sim_map@100`
- `best_epoch`
- `best_dev_metric`

Epoch 0 uses `global_step=0` and `train_loss=null`.

Alternative considered: add full dev retrieval metrics per epoch. Rejected because the user explicitly chose the minimal set, and full metrics would expand runtime and reporting scope.

## Risks / Trade-offs

- [Loss hook depends on PyTorch module hook behavior] -> Keep it local to dense-ft, test it with fake loss modules, and fall back to a clear error if the configured loss object cannot expose forward outputs.
- [Callback epoch numbering is zero-based in SentenceTransformers] -> Normalize rows to one-based training epochs and reserve epoch 0 for the baseline.
- [Manual save may write the base model when training does not improve] -> This is intended; it makes `best_model` truthful under the selected metric.
- [No step-level diagnostics] -> This is acceptable for the current goal of overfit/underfit diagnosis across epochs, not optimizer debugging inside an epoch.

## Migration Plan

1. Add dense-ft tests for epoch-0 metric rows, epoch-end rows, train-loss accumulation, best-epoch tracking, and best-model save behavior.
2. Implement an internal dense-ft training metrics tracker and loss observer in `graph_memory.models.dense_finetune.training`.
3. Update the SentenceTransformers 2.7.0 fit adapter to evaluate the baseline, pass an epoch callback, manually save best model, and return metric records.
4. Remove the final-only `trainer.evaluate()` row path from dense-ft orchestration while keeping the generic `DenseFinetuneTrainingResult.metric_records` contract.
5. Run focused dense-ft tests and strict OpenSpec validation.

Rollback is a normal code revert. Existing dense-ft run artifacts remain readable; older `train_metrics.jsonl` files simply have the old single-row shape.

## Open Questions

None. The metric set is intentionally limited to epoch-level selected dev metric, train loss, global step, and best-epoch tracking.

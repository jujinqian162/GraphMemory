## Context

`learned_graph_rgcn_retriever` currently trains with node-level BCE rank loss, optional learned-edge BCE, and sparse gate penalty. The model already receives same-task positive and negative samples through `TrainingBatch`, including `sample_task_ids`, `sample_node_ids`, `sample_types`, and `labels`.

The current loss can learn that gold evidence nodes are positive without directly penalizing a same-question hard negative that scores above a gold node. Recent 2Wiki results show this is the active failure shape: proposal graphs include the gold evidence nodes, but top-k output can still miss full support because one gold node is outranked by distractors.

## Goals / Non-Goals

**Goals:**

- Add a configurable task-local pairwise ranking loss for R-GCN training.
- Make gold evidence samples score above same-task negative samples, with hard negative sample types weighted more than easy random negatives.
- Preserve existing BCE, edge loss, and sparse gate loss behavior.
- Enable the new pairwise loss for `learned_graph_rgcn_retriever` by default.
- Keep existing R-GCN methods compatible by defaulting pairwise loss to disabled unless their config opts in.
- Record pairwise loss values and weights in training metrics and checkpoint metadata.

**Non-Goals:**

- Do not change proposal graph construction or edge vocabulary.
- Do not add a new retrieval method id.
- Do not change evaluation metrics or top-k retrieval output format.
- Do not add joint text encoder fine-tuning.
- Do not remove BCE rank loss in this change.

## Decisions

### Decision 1: Use weighted pairwise logistic loss

For each training batch, group samples by `task_id`. For every same-task positive sample `p` and negative sample `n`, compute:

```text
softplus(-(score[p] - score[n]) / temperature) * weight(sample_type[n])
```

The loss is the weighted mean over valid pairs. This directly penalizes negatives ranked above gold evidence and keeps providing smooth gradients when scores are close.

Alternative considered: margin ranking loss. It stops producing gradients once the margin is satisfied, which is less aligned with top-heavy ranking pressure.

### Decision 2: Keep BCE as a stabilizing term

The existing BCE rank loss remains available and still defaults to `1.0` at the config-record level. The learned graph method config will reduce it to `0.5` and set pairwise loss to `1.0`, making pairwise ranking the main ordering signal while preserving binary classification calibration.

Alternative considered: replace BCE entirely. That would increase regression risk for existing training behavior and make early training less stable.

### Decision 3: Configure negative type weights in method config

The loss config will add:

- `pairwise_rank_loss_weight`
- `pairwise_temperature`
- `pairwise_negative_type_weights`

Default config-record values keep pairwise disabled. The learned graph method config opts in with weights for `easy_random`, `hard_bm25`, `hard_dense`, and `hard_graph_neighbor`.

Alternative considered: hard-code negative weights in training code. That would hide experiment behavior outside method config and make ablations harder.

### Decision 4: Compute pairwise loss from sampled training pairs first

The first implementation uses the existing sampled positives and negatives in each `TrainingBatch`. It does not require full-ranking batches during training and does not change pair sampling artifacts.

Alternative considered: compute listwise loss over every candidate node in every task. That is closer to evaluation but would require much larger batches and a broader training-data contract change.

## Risks / Trade-offs

- Pairwise loss can overweight tasks with many sampled negatives -> Normalize by total pair weight rather than raw pair count.
- Hard-negative weights can overfit to sampled distractors -> Keep weights configurable and record them in metrics/checkpoints.
- Existing tests may assume loss component shape -> Add pairwise fields without removing current rank/edge/sparse metrics.
- Batch grouping by task can be inefficient if implemented naively -> Use simple per-batch grouping first; current batch sizes are moderate and correctness matters more than micro-optimization.

## Migration Plan

1. Add failing tests for pairwise loss computation and config defaults.
2. Extend loss config records and validation.
3. Implement weighted pairwise logistic loss in R-GCN training.
4. Record pairwise metrics and config values.
5. Enable pairwise loss in `learned_graph_rgcn_retriever` config and docs.
6. Validate focused tests, type checking, and OpenSpec.

Rollback is additive: set `pairwise_rank_loss_weight` to `0.0` in method config to recover previous training behavior without changing model architecture or retrieval code.

## Open Questions

None for the first implementation. Top-M negative filtering and listwise losses should be evaluated after the pairwise baseline has a formal run.

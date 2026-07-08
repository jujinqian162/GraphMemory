## 1. Pairwise Loss Tests

- [x] 1.1 Add failing tests for R-GCN loss config pairwise defaults and supported negative type weights.
- [x] 1.2 Add failing tests for same-task weighted pairwise logistic loss, cross-task pair exclusion, and weighted mean normalization.
- [x] 1.3 Add failing training metrics test for pairwise diagnostics.

## 2. Config and Validation

- [x] 2.1 Extend R-GCN loss config records with pairwise ranking loss fields.
- [x] 2.2 Extend R-GCN training config validation for positive temperature, non-negative weights, and supported negative sample types.

## 3. Training Loss Implementation

- [x] 3.1 Compute task-local weighted pairwise ranking loss from training batch logits, labels, task ids, and sample types.
- [x] 3.2 Add pairwise ranking loss to total R-GCN loss according to `pairwise_rank_loss_weight`.
- [x] 3.3 Record pairwise loss, pair count, weight, temperature, and negative type weights in training metrics.

## 4. Method Config and Documentation

- [x] 4.1 Enable pairwise ranking loss in `configs/methods/learned_graph_rgcn_retriever.json`.
- [x] 4.2 Update method documentation to describe pairwise ranking loss fields and objective.

## 5. Verification

- [x] 5.1 Run focused pairwise/R-GCN loss tests outside the Windows sandbox.
- [x] 5.2 Run `uv run basedpyright --level error` outside the Windows sandbox.
- [x] 5.3 Run `openspec validate add-pairwise-rgcn-ranking-loss --strict`.

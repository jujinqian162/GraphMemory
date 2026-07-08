## Why

The current trainable R-GCN objective treats evidence ranking as independent node classification. Recent 2Wiki learned-graph results show the gold evidence nodes are present in the candidate graph, but misses are still top-k ordering failures where one gold node is outranked by hard distractors.

This change adds task-local pairwise ranking supervision so gold evidence nodes are trained to score above same-question negatives, improving top-heavy ranking behavior rather than only learning that gold nodes are positive.

## What Changes

- Add configurable pairwise ranking loss fields to R-GCN training loss config.
- Compute a task-local weighted pairwise logistic loss between gold evidence samples and same-task negative samples.
- Weight hard negatives more heavily than easy random negatives, using method-configurable negative sample type weights.
- Record pairwise loss values and configuration in train metrics and checkpoint metadata.
- Enable the new loss for `learned_graph_rgcn_retriever` by default while preserving existing R-GCN behavior unless configured.
- Keep proposal graph construction, method IDs, checkpoint-backed retrieval, edge loss, and sparse gate loss semantics unchanged.

## Capabilities

### New Capabilities

- `rgcn-pairwise-ranking-loss`: Configurable task-local pairwise ranking loss for trainable R-GCN evidence retrievers.

### Modified Capabilities

- None.

## Impact

- Affected config/model code: R-GCN loss config records, validation, method config parsing, and checkpoint metadata.
- Affected training code: R-GCN loss computation, task-local grouping of sampled logits, loss component accounting, and metric records.
- Affected method configs/docs: `learned_graph_rgcn_retriever` loss defaults and method documentation.
- Affected tests: focused training-loss tests, method config tests, and OpenSpec validation.

## Why

The current Ours R-GCN retriever learns node ranking on a graph whose edges are fixed by `build_graph`. If that fixed graph misses important evidence paths, the model cannot recover them during training.

This change introduces a first-class `learned_graph_rgcn_retriever` that uses a high-recall proposal graph and learns edge gates, edge weights, and evidence ranking jointly while preserving the existing request-first and checkpoint-backed R-GCN boundaries.

## What Changes

- Add `learned_graph_rgcn_retriever` as a public retrieval method, separate from `dense_rgcn_graph_retriever` and `dense_ft_rgcn_graph_retriever`.
- Add a method-specific proposal graph stage owned by the new method's workflow dependency chain.
- Add configurable R-GCN loss weights with defaults `rank_loss_weight=1.0`, `edge_loss_weight=0.2`, and `sparse_loss_weight=0.05`.
- Extend R-GCN batching/model/training so the new method can learn edge gates and compute rank, edge, and sparse losses.
- Reuse existing dataset projectors, train-pair generation, checkpoint-backed retrieval, and path-metric evaluation.
- Adapt both HotpotQA and 2Wiki experiment configs to select the new method without changing existing method behavior.
- Preserve `gold_dependency_edges` as label-only supervision/evaluation data; they must not enter proposal graph construction or test-time graph tensors.

## Capabilities

### New Capabilities

- `learned-graph-rgcn-retriever`: Public trainable graph retriever with method-local proposal graphs, learned edge gating, configurable rank/edge/sparse losses, checkpoint-backed inference, and HotpotQA/2Wiki workflow integration.

### Modified Capabilities

- None.

## Impact

- Affected registry/config code: retrieval method IDs, method definitions, method config parsing, and stage config resolution.
- Affected workflow code: manifest paths, planner dependencies, and stage config generation for method-local proposal graphs.
- Affected graph code: proposal graph configuration and a proposal graph build CLI that reuses existing graph construction rules.
- Affected model code: R-GCN batching, tensorization, neural model factory, training metrics, checkpoint metadata, and inference loading.
- Affected experiment configs and docs: HotpotQA and 2Wiki named configs, method config file, method list, and run commands.
- Affected tests: registry, workflow planning, proposal graph leakage, batching, model behavior, training loss, checkpoint method identity, and dataset-specific workflow smoke planning.

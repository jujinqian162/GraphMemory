## Context

The existing Ours path, `dense_rgcn_graph_retriever`, trains an R-GCN evidence scorer over a graph artifact produced by the current graph builder. The model learns input projection, relation-aware message passing, and evidence scoring, but graph structure and graph edge weights are fixed before training.

This is a hard ceiling when `build_graph` omits useful evidence links. The new method needs the graph builder to act as a high-recall proposal mechanism and leave final edge usefulness to a learned edge gate. The implementation must still preserve the repo's request-first boundary: dataset records project into consumer requests, retrievers consume those requests/artifacts, and label-only fields never become input-visible graph data.

The current workflow also treats method dependency chains as method-owned. Therefore the new method must own its proposal graph stage instead of depending on another method's shared `graphs` stage.

## Goals / Non-Goals

**Goals:**

- Add `learned_graph_rgcn_retriever` as a registry-visible public method.
- Use method-local high-recall proposal graph artifacts for the new method.
- Learn edge gates and use them to scale R-GCN message edge weights.
- Train with configurable rank, edge, and sparse loss weights defaulting to `1.0`, `0.2`, and `0.05`.
- Reuse existing graph construction rules, train pair generation, R-GCN internals, checkpoint-backed retrieval, and evaluation surfaces where practical.
- Adapt both HotpotQA and 2Wiki workflows while preserving leakage-safe dataset boundaries.

**Non-Goals:**

- Do not replace or silently change `dense_rgcn_graph_retriever` or `dense_ft_rgcn_graph_retriever`.
- Do not add joint encoder fine-tuning in this change.
- Do not add a second checkpoint graph retriever adapter.
- Do not insert `gold_dependency_edges`, supporting facts, answer strings, or evidence triples into proposal graph construction.
- Do not run a broad hyperparameter search for the three loss weights.

## Decisions

### Decision 1: Model the retriever as a separate public method

`learned_graph_rgcn_retriever` will be a distinct `RetrievalMethodId` and `MethodDefinition`. This keeps result tables and experiment configs explicit and avoids hiding a materially different architecture behind a config switch on the existing R-GCN method.

Alternative considered: add a `learned_edges` flag to `dense_rgcn_graph_retriever`. That would make comparison and workflow dependencies ambiguous because the graph artifact semantics differ.

### Decision 2: Use method-local proposal graphs

The new method will introduce a `proposal_graphs` stage in its own dependency chain:

```text
prepare -> proposal_graphs -> pairs/train/retrieve -> evaluate/aggregate
```

Existing graph methods keep their own chain:

```text
prepare -> graphs -> pairs/train/retrieve -> evaluate/aggregate
```

The chains share prepared dataset records and dataset projectors, but not graph artifact semantics.

Alternative considered: build proposal graphs from existing graph artifacts. That would preserve the same recall ceiling that this change is intended to remove.

### Decision 3: Reuse existing graph rules with higher proposal caps first

The first implementation will keep the current edge vocabulary and reuse the existing graph builder rules with higher caps for query overlap, entity neighbors, and bridge edges. A new semantic candidate edge type is deferred until high-recall caps plus learned gates have been evaluated.

Alternative considered: add embedding-only semantic candidate edges immediately. That would increase tensorization, relation-vocabulary, and ablation scope before the simpler proposal-gate design is tested.

### Decision 4: Keep node ranking as the primary loss

Training will optimize:

```text
L = rank_loss_weight * L_rank
  + edge_loss_weight * L_edge
  + sparse_loss_weight * L_sparse
```

`L_rank` is the existing node-level BCE loss. `L_edge` is an auxiliary BCE over candidate edges with label-side supervision when available. `L_sparse` is a mean gate penalty that discourages turning on every proposal edge.

Alternative considered: rely on node BCE alone. That does not directly penalize dense gate usage and can learn shortcuts that preserve ranking while leaving the graph too noisy for path metrics and interpretability.

### Decision 5: Keep `gold_dependency_edges` label-only

2Wiki `gold_dependency_edges` can label candidate proposal edges for training and can be used by path metrics. They MUST NOT be used to add proposal edges or to populate train-time/test-time graph tensors directly. HotpotQA has no edge labels, so edge loss sample count should be zero without failing training.

Alternative considered: add missing gold edges into the graph to improve path metrics. That leaks labels into graph construction and invalidates retrieval evaluation.

### Decision 6: Reuse checkpoint-backed retrieval

The new method should continue through the existing checkpoint graph retriever loader and provenance path. Checkpoints must carry method identity, model config, and loss config so old and new R-GCN checkpoints reject each other when loaded with the wrong method.

Alternative considered: implement a separate learned-graph retriever adapter. That would duplicate established lifecycle behavior and increase future maintenance cost.

## Risks / Trade-offs

- Proposal graphs can become too dense -> Use fixed high-recall caps, sparse loss, edge/gate metrics, and workflow tests that confirm the proposal stage is method-local.
- Edge supervision can be absent on HotpotQA -> Record `edge_loss_sample_count=0` and keep rank/sparse training valid.
- Edge label alignment can be partial on 2Wiki -> Label only candidate edges whose endpoints match `gold_dependency_edges`; do not patch missing gold edges into the proposal graph.
- Model output shape changes can break existing R-GCN paths -> Prefer adding a learned model/output path while preserving the existing model behavior, or update all R-GCN consumers in one patch with regression tests.
- The change spans registry, workflow, graph, model, and configs -> Implement task-by-task with focused failing tests before production edits.

## Migration Plan

1. Register the method and config types without changing existing methods.
2. Add proposal graph config/build/planner support for the new method.
3. Extend batching and model internals for learned edges.
4. Add loss computation, metrics, checkpoint metadata, and inference loading.
5. Wire HotpotQA and 2Wiki configs and docs.
6. Verify targeted tests, type checks, OpenSpec validation, and plan smokes before running any expensive full workflow.

Rollback is straightforward because the new method is additive. Removing the method from experiment configs disables the new path while leaving existing R-GCN behavior intact.

## Open Questions

- None for the first implementation. Semantic candidate edges and ablation method aliases should be considered only after the additive first version is verified.

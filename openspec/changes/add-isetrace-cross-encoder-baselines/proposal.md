## Why

The WSDM study still lacks a genuinely strong graph-free scorer. Flat Dense-FT is a bi-encoder and cannot model query-candidate token interactions, so reviewers can attribute the residual R-GCN result to a weak flat baseline rather than to candidate segmentation or graph structure.

## What Changes

- Add a trainable `cross_encoder` retrieval method with `variant=flat|provenance_unit`; retain `flat` as the default and restrict `provenance_unit` to ISETrace.
- Fine-tune one E5-base-v2 sequence-classification scorer over natural-query exact-span supervision with pointwise binary cross-entropy, shared hyperparameters, and task-local development Recall@5 checkpoint selection.
- Score every task-local candidate directly instead of introducing a Dense top-N bottleneck; neither variant receives a graph, relation, graph identifier feature, or provenance trace.
- Reuse the existing ISETrace candidate projections and sampled pair artifacts while recording method, candidate view, backbone, max length, training seed, selection rule, and source digests in cache/artifact/checkpoint identity.
- Add strict checkpoint validation, full-ranking retrieval, six-budget evaluation, smoke/config/cache tests, and operations documentation for seeds 13/17/29.

## Capabilities

### New Capabilities

- `isetrace-cross-encoder-retrieval`: Defines matched flat and provenance-unit Cross-Encoder baselines, graph isolation, training/checkpoint identity, and complete-candidate evaluation.

### Modified Capabilities

None.

## Impact

Affected surfaces include the closed retrieval method enum and config union; trainable profile settings; pair/model/rank workflow dispatch; Cross-Encoder training, metadata, and retrieval modules; model input provisioning; inspection and operations documentation; and focused config/workflow/retrieval tests. No graph schema, ISETrace split, authoring data, evaluation metric, Dense/Dense-FT behavior, or R-GCN lifecycle changes.

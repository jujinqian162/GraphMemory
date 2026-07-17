## Why

`execution_provenance_rgcn_retriever` is trainable and already participates in the normal experiment workflow, but it has no registered ablation suite and the planner rejects it when `ablation.enable=true`. This prevents the provenance R-GCN from producing controlled ablation runs through the repository's public experiment configuration.

## What Changes

- Register an execution-provenance R-GCN ablation suite containing only variants that change signals actually consumed by that model: `wo_graph`, `wo_edge_type`, `wo_edge_weight`, and `wo_hard_negatives`.
- Extend the provenance R-GCN model configuration and construction path so model-level ablations are explicit, checkpointed, and applied during training and retrieval.
- Generalize workflow variant patching so `execution_provenance_rgcn_retriever` receives isolated pair/train/retrieve/evaluate invocations and normal baseline aliases.
- Expose the suite through `experiment/inspect.py kind=ablations` and document a copyable Hydra command using `ablation.enable=true` and `ablation.variants='[...]'`.
- Add regression coverage for discovery, planning, patch semantics, artifact isolation, and model/tensorization behavior.

## Capabilities

### New Capabilities

- `execution-provenance-rgcn-ablations`: Defines the supported execution-provenance R-GCN variants, their concrete signal removal, and their experiment workflow behavior.

### Modified Capabilities

None.

## Impact

- Affected registry and planning code: `graph_memory/registry/ablations.py` and `graph_memory/experiment/planning.py`.
- Affected typed configuration and training path: experiment config models, the execution-provenance R-GCN method YAML, trainer construction, checkpointed model config, model message transforms, and provenance tensorization.
- Affected operator surface: ablation inspection and the 2Wiki provenance runbook.
- No public method identifier, dataset schema, or ordinary non-ablation workflow changes.

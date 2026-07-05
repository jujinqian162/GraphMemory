## Why

2Wiki results show that `learned_graph_rgcn_retriever` learns stronger graph structure than the current Dense-FT seeded R-GCN, but its final evidence ranking still trails because it uses the frozen base E5 encoder. The next improvement should combine learned proposal/gated graph structure with the existing Dense-FT seed encoder without adding another public method.

## What Changes

- Make `learned_graph_rgcn_retriever` use the workflow-produced `dense_ft` checkpoint as its R-GCN seed/text encoder.
- Keep the public method id as `learned_graph_rgcn_retriever`; do not add a new method or result-table row.
- When `learned_graph_rgcn_retriever` is selected, automatically include the `dense_ft` pair/train dependency needed to produce `learned/dense_ft/checkpoints/best_model`.
- Pass that Dense-FT model directory into the learned graph R-GCN train stage via the existing `seed_checkpoint` mechanism.
- Keep proposal graph artifacts method-local to `learned_graph_rgcn_retriever`; do not make them depend on the shared `graphs` stage.
- Preserve existing loss weights and learned edge-gate behavior.

## Capabilities

### New Capabilities

- `learned-graph-rgcn-dense-ft-seeding`: `learned_graph_rgcn_retriever` uses the run-local Dense-FT checkpoint as its seed encoder while keeping method identity, proposal graphs, and learned edge-gate training unchanged.

### Modified Capabilities

- None.

## Impact

- Affected workflow dependency code: train dependency expansion and stage planning for `learned_graph_rgcn_retriever`.
- Affected stage config code: learned graph R-GCN train config must include `io.seed_checkpoint` pointing to the Dense-FT model directory.
- Affected docs/config docs: learned graph method docs must state that Dense-FT is an implicit training dependency.
- Affected tests: workflow planning and stage-config tests for HotpotQA and 2Wiki must assert the new seed checkpoint dependency without adding a new public method.

## Why

2WikiMultiHopQA now has dataset-specific records, projectors, path labels, and a named tiny experiment, but the named workflow still stops before the trainable methods that matter for Phase 2 comparison. This change makes 2Wiki runnable for both `dense_ft` and the Ours `dense_rgcn_graph_retriever` path while preserving the request-first, leakage-safe boundary.

## What Changes

- Extend the 2Wiki named experiment workflow so `dense_ft` and `dense_rgcn_graph_retriever` are first-class selectable methods with method configs.
- Keep all trainable method devices on `cuda`; do not introduce CPU fallback profiles in this change.
- Use 2Wiki `supporting_facts` as node-level supervision for Dense-FT and R-GCN training.
- Keep 2Wiki `gold_dependency_edges` label-only and use them only for path-level evaluation, not graph construction or R-GCN loss.
- Preserve Dense-FT as a flat text retrieval baseline whose path metrics remain `N/A`.
- Preserve R-GCN as graph-aware Ours, reporting real path metrics only through the returned `retrieved_subgraph`.
- Add focused tests and workflow smoke coverage proving the 2Wiki trainable path is planned, configured, and executable.

## Capabilities

### New Capabilities

- `twowiki-trainable-methods`: 2Wiki trainable-method workflow support for Dense-FT and R-GCN, including CUDA-only trainable configs, leakage-safe supervision, path metric semantics, and workflow verification.

### Modified Capabilities

- None.

## Impact

- Affected configs: `configs/experiments/2wiki_tiny.json`, and possibly 2Wiki-specific method config files if separate trainable smoke configs are needed.
- Affected workflow code and tests: experiment manifest/stage config generation for 2Wiki trainable methods, train/retrieve/evaluate stage config tests, and named recipe tests.
- Affected validation: method-device assertions, `gold_dependency_edges` leakage boundaries, and path metric `N/A` versus numeric behavior.
- No new model architecture, R-GCN edge type, path-aware loss, or Dense-FT graph retrieval method is introduced.

## 1. Lock the regression contract

- [x] 1.1 Add ablation discovery and workflow-planning tests for the execution-provenance R-GCN suite, baseline aliases, selected variants, and variant-local artifact invalidation.
- [x] 1.2 Add provenance model and tensorization tests proving `wo_graph`, `wo_edge_type`, and `wo_edge_weight` alter the named computation while ordinary behavior remains unchanged.
- [x] 1.3 Add config/checkpoint regression coverage for the model ablation field, resolved policies, and loading existing full-model checkpoint payloads without the new fields.

## 2. Register and plan provenance ablations

- [x] 2.1 Register the four-variant `execution_provenance_rgcn_retriever` suite with truthful changed dimensions and invalidation stages.
- [x] 2.2 Add the required full-model ablation value to the typed method config and default method YAML.
- [x] 2.3 Generalize planner variant patching to execution-provenance R-GCN configs while preserving pair aliases and variant-local pair generation.

## 3. Apply model-level ablation semantics

- [x] 3.1 Add a provenance model-config constructor that resolves ablation names into checkpointed layer, relation-transform, and edge-weight policies.
- [x] 3.2 Make the provenance model consume typed/shared relation-transform policy and make tensorization consume artifact/uniform edge-weight policy.
- [x] 3.3 Route the trainer through the resolved provenance model config and preserve old full-model checkpoint loading.

## 4. Operator surface and verification

- [x] 4.1 Document the supported execution-provenance variants and a copyable `ablation.enable=true` plus `ablation.variants='[...]'` command.
- [x] 4.2 Run focused tests, OpenSpec validation, lint/type checks, and a real plan/inspect acceptance check for the requested CLI surface.

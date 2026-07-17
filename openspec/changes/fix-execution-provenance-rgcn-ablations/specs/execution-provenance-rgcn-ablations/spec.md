## ADDED Requirements

### Requirement: Provenance R-GCN exposes a truthful ablation suite
The system SHALL register `execution_provenance_rgcn_retriever` with the `full_rgcn` baseline alias and exactly the executable variants `wo_graph`, `wo_edge_type`, `wo_edge_weight`, and `wo_hard_negatives`. A registered variant MUST remove a signal that the execution-provenance R-GCN actually consumes.

#### Scenario: Inspect lists the provenance suite
- **WHEN** an operator runs `experiment/inspect.py kind=ablations`
- **THEN** the output includes `execution_provenance_rgcn_retriever`, its `full_rgcn` baseline alias, and the four executable variants with their changed dimensions

#### Scenario: Evidence-only variants are not advertised for provenance
- **WHEN** an operator inspects the execution-provenance R-GCN suite
- **THEN** variants for bridge, entity-overlap, sequential, query-overlap, or dense seed-score signals are absent because those signals are not consumed by this model

### Requirement: Public ablation configuration expands provenance runs
The experiment planner SHALL accept `ablation.enable=true` with a non-empty supported subset in `ablation.variants` when `execution_provenance_rgcn_retriever` is selected. It SHALL retain the ordinary method run, create one isolated variant workflow per selected variant, alias unchanged upstream artifacts, and include the ordinary baseline plus variants in ablation aggregation.

#### Scenario: Selected provenance variants are planned
- **WHEN** the selected method is `execution_provenance_rgcn_retriever` and overrides include `ablation.enable=true` and `ablation.variants='[wo_graph,wo_edge_type]'`
- **THEN** the plan contains ordinary provenance invocations, isolated train/retrieve/evaluate invocations for both variants, pair aliases for both variants, and aggregate selections for `full_rgcn`, `wo_graph`, and `wo_edge_type`

#### Scenario: Pair-changing provenance variant is planned
- **WHEN** `wo_hard_negatives` is selected for `execution_provenance_rgcn_retriever`
- **THEN** the planner creates variant-local pairs, train, retrieve, and evaluate invocations instead of aliasing the ordinary pair artifact

#### Scenario: Unsupported provenance-only selection fails clearly
- **WHEN** only `execution_provenance_rgcn_retriever` is selected with an ablation variant outside its registered suite
- **THEN** planning fails before execution and identifies the unsupported variant

### Requirement: Each provenance model ablation changes its named signal
The provenance R-GCN training configuration and checkpointed model configuration SHALL record the effective model ablation. `wo_graph` SHALL use zero message-passing layers, `wo_edge_type` SHALL share one message transform across relations, and `wo_edge_weight` SHALL replace artifact edge weights with uniform weights while preserving graph topology.

#### Scenario: Graph propagation is removed
- **WHEN** a `wo_graph` provenance model is constructed
- **THEN** its graph encoder contains zero R-GCN layers and candidate scoring still operates on projected node and query states

#### Scenario: Relation-specific transforms are removed
- **WHEN** a `wo_edge_type` provenance model is constructed
- **THEN** every message edge uses the same learned linear transform regardless of relation identifier

#### Scenario: Artifact edge weights are removed
- **WHEN** a provenance request is tensorized under `wo_edge_weight`
- **THEN** every visible message edge has weight `1.0` while edge endpoints and relation identifiers remain unchanged

### Requirement: Hard-negative ablation invalidates pair generation
The `wo_hard_negatives` provenance variant SHALL set all hard-negative sampling counts to zero, SHALL preserve easy random negatives, and SHALL invalidate the workflow beginning at the pair stage.

#### Scenario: Hard-negative counts are removed
- **WHEN** the planner applies `wo_hard_negatives` to `execution_provenance_rgcn_retriever`
- **THEN** the variant pair configuration has zero BM25, dense, and graph-neighbor hard negatives while retaining the baseline easy-random count

### Requirement: Ordinary provenance training remains stable
When ablation is disabled, the system SHALL preserve the existing execution-provenance R-GCN plan and full-model semantics.

#### Scenario: Ablation disabled
- **WHEN** `ablation.enable=false`
- **THEN** no provenance variant invocations or ablation table are planned and the model uses typed relation transforms, artifact edge weights, and the configured positive number of graph layers

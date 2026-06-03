## ADDED Requirements

### Requirement: Graph-rerank scoring is domain-owned
The system SHALL move graph-rerank config parsing, scoring components, candidate expansion, normalization, debug helpers, and method adapter into retrieval graph-rerank modules without changing scoring behavior.

#### Scenario: Graph-rerank ranking remains stable
- **WHEN** graph-rerank retrieval runs on the frozen tiny fixture with the same config
- **THEN** candidate nodes, component contributions, ranking order, scores, tie-breaks, and retrieved subgraph remain equivalent to the pre-refactor behavior

#### Scenario: Graph-rerank config validation remains stable
- **WHEN** graph-rerank config records are parsed and validated
- **THEN** defaults, unsupported-field failures, deprecated `type_weights` failures, and finite-number checks remain equivalent to the pre-refactor behavior

### Requirement: Graph-rerank tuning is domain-owned
The system SHALL move graph-rerank grid generation, initial-score caching usage, objective scoring, best-config selection, and tuning service into retrieval tuning modules without changing tuning semantics.

#### Scenario: Tuning selected config remains stable
- **WHEN** graph-rerank tuning runs with the same fixture, grid, labels, graphs, and dense seed encoder
- **THEN** candidate rows and selected config remain equivalent to the pre-refactor behavior

#### Scenario: Tuning objective and tie-breaks remain stable
- **WHEN** candidate rows have comparable aggregate metrics
- **THEN** best-config selection uses the same objective, Full Support@10 tie-break, latency tie-break, and retrieved-edge tie-break as before

### Requirement: Old graph-rerank root modules are not kept as broad facades
The system SHALL remove old root graph-rerank and tuning modules from production/script/test imports after their owned functionality has moved.

#### Scenario: Residual old imports are absent
- **WHEN** repository imports are scanned after Change B
- **THEN** production, script, and test code no longer imports `graph_memory.rerank`, `graph_memory.rerank_config`, or `graph_memory.tuning`

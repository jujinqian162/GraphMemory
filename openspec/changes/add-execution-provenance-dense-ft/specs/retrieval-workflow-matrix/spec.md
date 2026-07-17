## MODIFIED Requirements

### Requirement: Execution provenance method matrix
The execution-provenance profile SHALL contain BM25, Dense, Dense-FT, GraphRAG, Execution-Provenance Retriever, and Execution-Provenance R-GCN. Both evidence R-GCN methods SHALL remain outside this family.

#### Scenario: Validate provenance family support
- **WHEN** Registry queries execution-provenance family capabilities
- **THEN** it returns the six target methods including `dense_ft` and excluding both evidence-only R-GCN methods

### Requirement: Conditional evidence graph workflow
Workflow SHALL construct EvidenceGraph only for evidence R-GCN, evidence-family pair/training, or explicit evidence-only evaluation needs. Dense-FT on execution provenance and the provenance R-GCN SHALL consume dataset-projected requests in the existing pair/train/retrieve stages without scheduling EvidenceGraph construction or introducing a new stage type.

#### Scenario: Provenance Dense-FT run
- **WHEN** a run selects `dense_ft` on `twowiki_provenance`
- **THEN** planner schedules prepare, text-only pairs, train, retrieve, evaluate, and aggregate with no EvidenceGraph construction or graph artifact dependency

#### Scenario: Provenance R-GCN run
- **WHEN** a run selects `execution_provenance_rgcn_retriever` on `twowiki_provenance`
- **THEN** planner retains its existing prepare, text-only pairs, train, retrieve, evaluate, and aggregate lifecycle with no EvidenceGraph construction solely for that method

#### Scenario: Evidence Dense-FT run
- **WHEN** a run selects `dense_ft` on a supported evidence dataset
- **THEN** planner retains EvidenceGraph construction for its configured graph-neighbor pair sampling and leaves the downstream Dense-FT lifecycle unchanged

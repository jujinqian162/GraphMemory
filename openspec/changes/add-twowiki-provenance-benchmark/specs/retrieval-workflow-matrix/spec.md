## MODIFIED Requirements

### Requirement: Execution provenance method matrix
The execution-provenance profile SHALL contain BM25, Dense, GraphRAG, Execution-Provenance Retriever, and Execution-Provenance R-GCN; Dense-FT and both evidence R-GCN methods SHALL remain outside this family.

#### Scenario: Validate provenance family support
- **WHEN** Registry queries execution-provenance family capabilities
- **THEN** it returns the five target methods including the new trainable provenance R-GCN and excluding evidence-only methods

### Requirement: Conditional evidence graph workflow
Workflow SHALL construct EvidenceGraph only for evidence R-GCN, evidence pair/training, or explicit evidence-only evaluation needs; the provenance R-GCN SHALL consume dataset-projected ExecutionProvenanceRankingRequest in the existing pair/train/retrieve stages without scheduling EvidenceGraph construction or introducing a new stage type.

#### Scenario: Provenance R-GCN run
- **WHEN** a run selects `execution_provenance_rgcn_retriever` on `twowiki_provenance`
- **THEN** planner schedules existing prepare, pair, train, retrieve, evaluate, and aggregate lifecycle stages as required and schedules no EvidenceGraph construction solely for that method

#### Scenario: Existing evidence R-GCN run
- **WHEN** a run selects either existing evidence R-GCN on a supported evidence dataset
- **THEN** planner retains the current EvidenceGraph construction and node-wise training/inference path


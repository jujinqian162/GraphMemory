## MODIFIED Requirements

### Requirement: Semantic input metadata
Every MethodDefinition SHALL explicitly declare request type, supported task families, required artifact, ranked-node/native-trace/trainable capabilities, and train/pair lifecycle needs. `dense_ft` SHALL declare `TextRankingRequest`, both evidence-retrieval and execution-provenance families, no graph artifact, a model-directory train artifact, and no native-edge trace capability. The provenance R-GCN SHALL continue to declare ExecutionProvenanceRankingRequest, execution-provenance family, no EvidenceGraph artifact, and trainable path capabilities.

#### Scenario: Inspect Dense-FT definition
- **WHEN** a caller inspects `dense_ft`
- **THEN** the definition declares flat text requests, both supported families, no graph artifact, Dense-Finetune lifecycle, and the existing `best_model` directory artifact

#### Scenario: Inspect evidence R-GCN definition
- **WHEN** a caller inspects either existing evidence R-GCN
- **THEN** the definition still declares EvidenceGraph request, evidence family, EvidenceGraph artifact, and node-wise behavior

#### Scenario: Inspect provenance R-GCN definition
- **WHEN** a caller inspects `execution_provenance_rgcn_retriever`
- **THEN** the definition remains Dense-seeded and declares ExecutionProvenanceRankingRequest, execution-provenance family, provenance checkpoint/config, train/pair lifecycle, and logical path metrics

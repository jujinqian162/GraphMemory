## MODIFIED Requirements

### Requirement: Exact public method registry
Registry SHALL expose exactly `bm25`, `dense`, `dense_ft`, `graphrag`, `dense_rgcn_graph_retriever`, `dense_ft_rgcn_graph_retriever`, `execution_provenance_retriever`, and `execution_provenance_rgcn_retriever` as public method IDs.

#### Scenario: Enumerate public methods
- **WHEN** a caller enumerates current public retrieval methods
- **THEN** the result equals the eight target IDs and contains no Memory Stream or retired graph-rerank ID

### Requirement: Semantic input metadata
Every MethodDefinition SHALL explicitly declare request type, supported task families, required artifact, ranked-node/native-trace/trainable capabilities, and train/pair lifecycle needs; the provenance R-GCN SHALL declare ExecutionProvenanceRankingRequest, execution-provenance family, no EvidenceGraph artifact, and trainable path capabilities.

#### Scenario: Inspect evidence R-GCN definition
- **WHEN** a caller inspects either existing evidence R-GCN
- **THEN** the definition still declares EvidenceGraph request, evidence family, EvidenceGraph artifact, and node-wise behavior

#### Scenario: Inspect provenance R-GCN definition
- **WHEN** a caller inspects `execution_provenance_rgcn_retriever`
- **THEN** the definition declares ExecutionProvenanceRankingRequest, execution-provenance family, provenance checkpoint/config, train/pair lifecycle, and logical path metrics

### Requirement: Pre-execution compatibility validation
Registry builders MUST validate concrete payload, request type, task family, required artifact, checkpoint family, and dataset capability before loading or running a method.

#### Scenario: Provenance request sent to evidence R-GCN
- **WHEN** an execution-provenance request is routed to either evidence R-GCN builder
- **THEN** the builder rejects it before checkpoint loading

### Requirement: Concrete build payloads
The system SHALL use flat, GraphRAG, Evidence R-GCN, stateless ExecutionProvenance, and trainable ExecutionProvenance R-GCN concrete build payloads, and MUST NOT use a generic graph payload to bridge evidence and provenance inputs.

#### Scenario: Wrong provenance payload class
- **WHEN** the provenance R-GCN builder receives an EvidenceGraph payload or stateless method config
- **THEN** the builder reports an explicit payload mismatch instead of adapting it implicitly

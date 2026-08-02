## MODIFIED Requirements

### Requirement: Method compatibility is validated by retrieval task family
The workflow SHALL classify ISETrace query tasks as `execution_provenance`. BM25, Dense, and GraphRAG SHALL accept text-derived execution-provenance requests without receiving a provenance graph. `provenance_path` SHALL remain a training-free method that requires a typed request carrying the query-independent physical provenance graph. `provenance_rgcn` SHALL support only execution provenance, SHALL require train/dev/test lifecycle inputs and a strict current checkpoint, and SHALL consume the same query-independent physical provenance graph through its provenance tensorizer. Evidence R-GCN methods SHALL retain their evidence-retrieval boundary.

#### Scenario: Flat method runs on ISETrace
- **WHEN** BM25, Dense, or GraphRAG is configured for ISETrace
- **THEN** the builder receives only text-ranking requests
- **AND** the workflow schedules no graph-training input

#### Scenario: Training-free provenance method lacks a graph
- **WHEN** `provenance_path` is built without the referenced query-independent provenance graph
- **THEN** request construction fails before ranking

#### Scenario: Trainable provenance method lacks lifecycle inputs
- **WHEN** `provenance_rgcn` is configured without prepared train, dev, or test inputs
- **THEN** workflow validation fails before pair construction or training

#### Scenario: Provenance method is used on an evidence dataset
- **WHEN** `provenance_path` or `provenance_rgcn` is composed with HotpotQA, 2Wiki, or MuSiQue
- **THEN** compatibility validation rejects the configuration

#### Scenario: Evidence R-GCN is used on ISETrace
- **WHEN** an evidence R-GCN method is composed with ISETrace
- **THEN** compatibility validation rejects the configuration rather than converting `ProvenanceGraph` to `EvidenceGraph`

### Requirement: Evaluation graphs are not method inputs by default
The workflow SHALL keep ISETrace exact-span labels separate from every retrieval method and MUST NOT treat the input provenance graph as gold dependency annotation. BM25, Dense, and GraphRAG MUST NOT receive the physical provenance graph. Only `provenance_path` and `provenance_rgcn` SHALL receive it through their typed execution-provenance boundaries. All methods SHALL preserve aligned natural-test task IDs, candidate IDs, source spans, and labels.

#### Scenario: Flat and provenance methods share natural test labels
- **WHEN** BM25, Dense, GraphRAG, `provenance_path`, and `provenance_rgcn` are evaluated on the same resolved natural test split
- **THEN** prediction task IDs, candidate IDs, exact source spans, and labels align across methods
- **AND** only the two provenance methods have graph access during retrieval

#### Scenario: Physical edges have no independent labels
- **WHEN** a method retrieves over an ISETrace graph whose natural records contain only exact gold spans
- **THEN** the workflow does not derive gold path or edge labels from that input graph
- **AND** path and edge accuracy remain unavailable

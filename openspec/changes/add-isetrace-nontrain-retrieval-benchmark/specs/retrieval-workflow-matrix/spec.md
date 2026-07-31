## MODIFIED Requirements

### Requirement: Method compatibility is validated by retrieval task family
The workflow SHALL classify ISETrace query tasks as `execution_provenance`. BM25, Dense, and GraphRAG SHALL accept text-derived execution-provenance requests. `provenance_path` SHALL require a typed provenance-path request and SHALL support only execution provenance. Evidence R-GCN methods SHALL retain their existing evidence-retrieval boundary.

#### Scenario: Flat method runs on ISETrace
- **WHEN** BM25 or Dense is configured for ISETrace
- **THEN** the builder receives only text-ranking requests and schedules no graph-training input

#### Scenario: Provenance method lacks a provenance graph
- **WHEN** `provenance_path` is built without the referenced query-independent graph
- **THEN** request construction fails before ranking

#### Scenario: Provenance method is used on an evidence dataset
- **WHEN** `provenance_path` is composed with HotpotQA, 2Wiki, or MuSiQue
- **THEN** compatibility validation rejects the configuration

### Requirement: Evaluation graphs are not method inputs by default
The workflow SHALL build the same output-only logical dependency graph for execution-provenance evaluation of every method. BM25, Dense, and GraphRAG MUST NOT receive that graph as retrieval input. Only `provenance_path` receives the full physical provenance graph.

#### Scenario: Flat and graph methods share evaluation labels
- **WHEN** four non-training methods run over the same prepared ISETrace split
- **THEN** prediction task IDs, candidate IDs, labels, and evaluation graph digests align across methods

### Requirement: Query-independent graphs do not claim query connectivity
Execution-provenance evaluation SHALL mark query-to-evidence connectivity unsupported when graphs contain no query-conditioned edges. It SHALL continue to compute evidence recall, complete support, evidence connectivity, path recall, and edge metrics where labels provide dependencies.

#### Scenario: ISETrace evaluation graph has no query edge
- **WHEN** an output-only provenance graph is evaluated
- **THEN** aggregate Query-Evidence Connectivity is `N/A` rather than a fabricated zero-quality claim

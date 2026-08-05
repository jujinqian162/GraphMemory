## ADDED Requirements

### Requirement: Retrieval execution uses preassembled method requests
The system SHALL run retrieval methods from execution tasks that already contain the exact method-family request to pass to the retrieval method.

#### Scenario: Flat retrieval receives text request
- **WHEN** retrieval execution runs a BM25, dense, or dense-ft method
- **THEN** it passes a `TextRankingRequest` directly to `RetrievalMethod.rank_task`

#### Scenario: Graph retrieval receives graph request
- **WHEN** retrieval execution runs a graph-rerank or R-GCN method
- **THEN** it passes a `GraphRankingRequest` containing candidates, graph, and initial scores directly to `RetrievalMethod.rank_task`

#### Scenario: Temporal retrieval receives temporal request
- **WHEN** retrieval execution runs a Memory Stream method
- **THEN** it passes a `TemporalMemoryRankingRequest` containing candidates, importance scores, and temporal metadata directly to `RetrievalMethod.rank_task`

### Requirement: Retrieval execution does not branch on concrete method classes
The system SHALL keep method-family request assembly outside `graph_memory.retrieval.execution.service`.

#### Scenario: Execution service has no concrete method dispatch
- **WHEN** the retrieval execution service is inspected
- **THEN** it does not import or check concrete method classes such as `GraphRerankMethod`, `MemoryStreamMethod`, or `TrainableGraphRetrievalMethod`

#### Scenario: Execution service does not compute method dependencies
- **WHEN** retrieval execution runs a graph-backed or temporal method
- **THEN** it does not compute seed rankings, seed signals, graph lookup fallbacks, or importance scores

### Requirement: Retrieval method request type is explicit
The system SHALL type `RetrievalMethod.rank_task` as accepting the supported method request union rather than an unconstrained object.

#### Scenario: Protocol rejects untyped request boundary
- **WHEN** the retrieval method protocol is inspected
- **THEN** the request parameter is typed as `TextRankingRequest | GraphRankingRequest | TemporalMemoryRankingRequest` or an equivalent named request union

### Requirement: R-GCN inference uses request graph authority
The system SHALL treat the graph attached to `GraphRankingRequest` as the authoritative graph for checkpoint-backed graph retriever inference.

#### Scenario: Request graph is used for tensorization
- **WHEN** `GraphRetrieverInference.rank_task` receives a `GraphRankingRequest`
- **THEN** full-ranking batches are built from `request.graph`

#### Scenario: Internal graph index does not override request graph
- **WHEN** a loader or method also has cached graphs for the same task id
- **THEN** those cached graphs do not replace the graph carried by the ranking request during that invocation

### Requirement: Request assembly remains at stage or registry boundaries
The system SHALL assemble method-family requests in dataset-aware stage or registry adapter code before calling retrieval execution.

#### Scenario: Graph rerank request assembly computes initial scores before execution
- **WHEN** a graph-rerank retrieval run is prepared
- **THEN** the stage or registry adapter computes seed initial scores from the text request and graph before creating the `GraphRankingRequest`

#### Scenario: R-GCN request assembly computes seed scores before execution
- **WHEN** a checkpoint-backed graph retrieval run is prepared
- **THEN** the stage or registry adapter computes seed scores using the configured seed signal provider before creating the `GraphRankingRequest`

#### Scenario: Memory Stream request assembly selects importance before execution
- **WHEN** a Memory Stream retrieval run is prepared
- **THEN** the stage or registry adapter selects and validates task importance records before creating the `TemporalMemoryRankingRequest`

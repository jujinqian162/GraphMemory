## ADDED Requirements

### Requirement: GraphRAG request carries method-owned graph
`GraphRAGRequest` SHALL contain a concrete `EntityKnowledgeGraph` assembled by the GraphRAG method package from the request's text candidates.

#### Scenario: GraphRAG graph is explicit but not dataset-owned
- **WHEN** a dataset supplies a TextRankingRequest to the GraphRAG builder
- **THEN** the builder creates the entity graph, places it in GraphRAGRequest, and leaves the dataset adapter free of GraphRAG-specific projection code

### Requirement: Request graph alignment
GraphRAG and execution-provenance requests SHALL validate task identity, candidate uniqueness, and graph-to-candidate references at construction or Registry build time.

#### Scenario: Reject missing GraphRAG candidate mapping
- **WHEN** an entity graph references a candidate ID absent from the GraphRAG request
- **THEN** request validation rejects the input before retrieval

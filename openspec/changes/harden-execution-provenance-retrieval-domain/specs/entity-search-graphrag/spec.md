## ADDED Requirements

### Requirement: Explicit GraphRAG request assembly
The GraphRAG builder SHALL deterministically assemble an `EntityKnowledgeGraph` from each `TextRankingRequest` and SHALL pass that graph in `GraphRAGRequest` before retrieval execution.

#### Scenario: Inspect assembled request
- **WHEN** Registry builds GraphRAG for a text request
- **THEN** the execution task contains a GraphRAG request with entities, relations, aliases, and candidate mappings derived without EvidenceGraph or provenance edges

### Requirement: FastGraphRAG-derived entity semantics
GraphRAG SHALL implement normalized entity mentions, unambiguous alias ownership, weighted co-occurrence relations, query entity linking, lexical and dense entity seeds, weighted PPR, and entity/relation projection to candidates.

#### Scenario: Alias-linked entity propagation
- **WHEN** a query uses an unambiguous alias for an entity connected to evidence in another candidate
- **THEN** the canonical entity receives query seed mass and PPR can raise the connected candidate score

#### Scenario: Relation contribution reaches candidate
- **WHEN** a candidate contains both endpoints of a high-scoring entity relation
- **THEN** the relation contribution is included in that candidate's projected score

### Requirement: GraphRAG search consumes an assembled graph
`GraphRAGMethod.rank_task` MUST search `request.knowledge_graph` and MUST NOT reconstruct the entity catalog or relation index inside the ranking lifecycle.

#### Scenario: Reuse assembled graph
- **WHEN** a preassembled GraphRAG request is ranked repeatedly
- **THEN** retrieval reuses the supplied entity graph and produces deterministic ranking and trace output

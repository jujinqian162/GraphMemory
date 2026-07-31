## MODIFIED Requirements

### Requirement: GraphRAG builds a method-private text-derived entity graph
GraphRAG SHALL derive entity evidence only from `TextRankingRequest` candidate text and input-visible text metadata. For document candidates with titles it SHALL retain typed title/body entity groups. When no title group exists, it SHALL deterministically form title-free groups from entities shared by at least two candidate texts. It MUST NOT consume an execution-provenance graph, graph labels, answer IDs, or support IDs.

#### Scenario: Document titles remain authoritative
- **WHEN** candidates expose valid title entities
- **THEN** GraphRAG uses the existing title/body grouping behavior rather than replacing it with title-free groups

#### Scenario: Trajectory outputs have no titles
- **WHEN** ToolOutput candidates contain a shared capitalized entity, URL, path, or filename but no title groups
- **THEN** GraphRAG may form a bounded shared-entity group from candidate text

#### Scenario: Provenance edge exists without textual entity overlap
- **WHEN** two candidates are linked only by a native provenance edge and share no extracted text entity
- **THEN** that edge cannot create a GraphRAG bridge

### Requirement: GraphRAG interventions remain bounded and abstaining
GraphRAG SHALL start from the frozen Dense ranking, suppress high-frequency entity hubs, resolve bounded partner proposals with the shared frozen encoder, preserve the configured Dense prefix, and return exact Dense output when no bridge causes a valid stable insertion.

#### Scenario: No shared entity survives gates
- **WHEN** no title or title-free group yields an accepted bridge
- **THEN** ranked IDs and scores are exactly Dense and the trace declares exact fallback

#### Scenario: Shared entity bridge is accepted
- **WHEN** a Dense anchor and lower-ranked partner share a non-hub entity and pass resolver/confidence gates
- **THEN** the partner may be stably inserted and the method emits one auditable `bridge_to` edge when both endpoints remain in top-k

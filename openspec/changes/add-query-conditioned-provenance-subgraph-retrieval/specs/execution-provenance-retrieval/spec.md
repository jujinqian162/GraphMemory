## ADDED Requirements

### Requirement: Default training-free provenance retrieval is joint subgraph retrieval
`execution_provenance_retriever` SHALL default to query-conditioned typed diffusion followed by budgeted connected-subgraph extraction while preserving the `ExecutionProvenanceRankingRequest` input, full candidate ranking output, public registry id, and non-trained status.

#### Scenario: Registry constructs the new default
- **WHEN** the method config specifies `execution_provenance_retriever` without a diagnostic variant override
- **THEN** the registry builds the query-conditioned PPR/Steiner configuration

#### Scenario: Legacy path mode is explicit
- **WHEN** a historical typed-beam or dependency-path diagnostic is requested
- **THEN** its variant and cache identity remain distinct from the default subgraph retriever

### Requirement: Full-graph connectors remain method-local
The retriever SHALL use native graph nodes outside the request candidate set only as internal transition and connectivity evidence. It MUST NOT add connector-only nodes to the ranked candidate list or shared retrieved-subgraph node surface.

#### Scenario: Tool call joins two candidate outputs
- **WHEN** a non-candidate tool call connects two selected candidate outputs
- **THEN** the ranked list contains only request candidates while the native trace records the tool-call connector

### Requirement: Frozen parameters participate in run identity
Every behavior-affecting diffusion, relation-description, transition, and connected-selection parameter SHALL participate in the resolved ranking config and cache identity.

#### Scenario: PPR damping changes cache key
- **WHEN** otherwise identical runs use different PPR damping values
- **THEN** they cannot reuse the same ranking artifact

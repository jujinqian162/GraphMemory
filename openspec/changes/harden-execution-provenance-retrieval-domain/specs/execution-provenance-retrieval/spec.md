## ADDED Requirements

### Requirement: Complete typed transition validation
`ExecutionProvenanceGraph` SHALL validate source and target node types for every public provenance edge type and MUST reject any transition not explicitly present in the schema.

#### Scenario: Reject nonsensical support edge
- **WHEN** an Answer node is connected to a ToolCall node with `supports`
- **THEN** graph construction fails before retrieval

### Requirement: Alternative provenance paths remain scoreable
Provenance expansion SHALL retain distinct bounded simple paths to the same candidate until path scoring and MUST NOT discard a path only because a shorter route reached that candidate first.

#### Scenario: Complete bound path competes with direct weak path
- **WHEN** one candidate is reachable by a direct `depends_on` edge and by a longer complete `feeds/returns` path with bindings
- **THEN** both paths are scored and the complete bound path can rank above the direct path

### Requirement: Score components are applied once
Path scoring SHALL calculate semantic relevance, binding consistency, provenance completeness, explicit grounding, path length penalty, and invalidation penalty once, and candidate projection MUST NOT reapply nested path components.

#### Scenario: Project selected path score
- **WHEN** a selected path contributes to several retrievable nodes
- **THEN** each node receives the selected path's final score once while unrelated candidates retain semantic fallback scores

### Requirement: Revision and lifecycle invalidation
Invalidation SHALL include targets of revision edges and nodes whose source metadata marks an invalid, invalidated, superseded, or obsolete lifecycle state.

#### Scenario: Metadata-invalidated support
- **WHEN** a support node has invalid lifecycle metadata without an explicit revision edge
- **THEN** paths containing that node receive the configured invalidation penalty

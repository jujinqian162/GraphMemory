## ADDED Requirements

### Requirement: Typed partner completion emits a closed native trace
The native trace union SHALL include a closed `typed_partner_completion` trace kind carrying dense ranks with original and final positions, the relation-description version, per-edge-type relation affinities, every partner proposal with its anchor, partner, path node ids, path edge types, confidence, acceptance flag and rejection reason, the protected prefix, connector node ids, emitted candidate edges, the scorer identity, and the exact-Dense fallback flag. Serialization and validation SHALL reject unknown fields.

#### Scenario: Round trip preserves every proposal
- **WHEN** a partner completion result is serialized and validated
- **THEN** all proposals including rejected ones survive with their reasons intact

#### Scenario: Relation affinities form a distribution
- **WHEN** the trace reports relation affinities
- **THEN** they cover exactly the edge types present in the graph and sum to one

#### Scenario: Connectors are non-candidates inside the graph
- **WHEN** the trace reports connector node ids
- **THEN** each is a native graph node and none is a request candidate

#### Scenario: Fallback consistency is enforced
- **WHEN** the trace declares exact Dense fallback
- **THEN** validation requires zero accepted proposals, zero connectors, and zero emitted edges

#### Scenario: Unknown field is rejected
- **WHEN** a serialized trace carries a field outside the closed schema
- **THEN** validation fails

## REMOVED Requirements

### Requirement: Query-conditioned subgraph native trace
**Reason**: The `execution_provenance_subgraph` trace describes PPR diffusion state, candidate prizes, and budgeted Steiner selection steps, none of which exist after the algorithm is replaced.

**Migration**: Replaced by the `typed_partner_completion` trace kind. Existing serialized artifacts remain readable from the archived `results/*-twp-sd13-v4` run directories.

### Requirement: Stateless provenance path native trace
**Reason**: The `execution_provenance_local` trace served the removed `typed_beam` and `dependency_path` variants and reports schema-gate outcomes that the replacement algorithm does not compute.

**Migration**: Replaced by the `typed_partner_completion` trace kind.

## ADDED Requirements

### Requirement: Build one query-independent graph per canonical trajectory
The system SHALL project a canonical trajectory into a provenance graph whose identity and contents depend only on the canonical trajectory and graph-builder configuration.

#### Scenario: Multiple queries over one trajectory
- **WHEN** different query texts or motif labels reference the same canonical trajectory
- **THEN** they reference the same graph ID and graph fingerprint
- **AND** no query text, query ID, answer, support ID, motif type, or template metadata appears in the graph serialization

### Requirement: Emit a minimal execution vocabulary
The core builder SHALL emit only the v1 core node kinds `execution.tool_call`, `execution.tool_output`, and `resource.artifact`, and the core relations `execution.returns`, `temporal.precedes`, `data.feeds`, `resource.reads`, and `resource.writes`.

#### Scenario: Native execution structure
- **WHEN** a canonical trajectory contains ordered paired tool calls and outputs
- **THEN** the graph contains one call and output node per execution, one `execution.returns` edge per pair, and deterministic `temporal.precedes` edges between consecutive calls

#### Scenario: Explicit artifact access
- **WHEN** a supported tool has an explicit path or URL argument
- **THEN** the graph creates a stable trajectory-local artifact node and the appropriate `resource.reads` or `resource.writes` edge

### Requirement: Derive feeds only from conservative exact bindings
The core builder SHALL emit `data.feeds` only for unique, high-information values found in an earlier output and reused in a later call argument.

#### Scenario: Unique path reuse
- **WHEN** an earlier output uniquely contains `/workspace/report.md` and a later call argument references the same path
- **THEN** the graph emits a deterministic `data.feeds` edge carrying binding type and value hash metadata

#### Scenario: Ambiguous or low-information value
- **WHEN** a value is produced by multiple outputs or is a boolean, null, short number, or generic short token
- **THEN** it does not create a `data.feeds` edge

### Requirement: Preserve extensible namespaced semantics
Graph node kinds and edge relations SHALL use validated namespaced identifiers and source spans so future annotation layers can add semantic nodes and relations without changing canonical source events.

#### Scenario: Future semantic extension
- **WHEN** a later annotator adds a `semantic.claim` node and `semantic.supports` edge with source-span references
- **THEN** the graph contract can represent them without pretending they were emitted by the core execution builder

#### Scenario: Invalid identifier or core endpoint
- **WHEN** a graph uses a malformed kind/relation identifier or a core relation with illegal endpoint kinds
- **THEN** graph validation fails before publication

### Requirement: Avoid semantic edge weights
The provenance graph SHALL NOT contain a query-conditioned or semantic edge-weight field.

#### Scenario: Serialized core edge
- **WHEN** a core edge is serialized
- **THEN** it records relation, endpoints, derivation, extractor identity, and optional JSON attributes but no ranking weight or query relevance score

## ADDED Requirements

### Requirement: Extract motif supervision without mutating the graph
The system SHALL derive motif specs from a query-independent provenance graph and store answer/support supervision outside the graph.

#### Scenario: Value-flow motif
- **WHEN** an output feeds a later call that has a returned output
- **THEN** the extractor can emit a `value_flow` motif whose support outputs and logical dependency are explicit in the motif spec
- **AND** the source graph remains byte-identical

#### Scenario: Artifact-lifecycle motif
- **WHEN** an earlier call writes an artifact and a later call reads the same artifact
- **THEN** the extractor can emit an `artifact_lifecycle` motif grounded in the paired call outputs and artifact dependency

#### Scenario: Composite motifs
- **WHEN** compatible value or artifact dependencies compose into a multi-hop path or multiple producers feed one consumer
- **THEN** the extractor can emit `multi_hop_flow` or `multi_source_join` motifs with complete support output IDs

### Requirement: Separate safe query slots from hidden labels
A motif spec SHALL distinguish public semantic slots available to a verbalizer from node IDs, answer values, binding hashes, and support labels.

#### Scenario: Render a query
- **WHEN** a template query is rendered from a motif
- **THEN** the renderer can access only declared safe slots
- **AND** undeclared hidden fields, graph IDs, node IDs, and answer values cannot be interpolated

### Requirement: Support multiple query intents per motif shape
The system SHALL model query intent separately from motif type so one graph structure can yield semantically different retrieval requests.

#### Scenario: One value-flow motif, different intents
- **WHEN** a value-flow motif supports upstream-producer, downstream-result, and complete-chain intents
- **THEN** the query synthesizer can emit distinct questions with intent-appropriate answer/support labels while referencing the same graph

### Requirement: Provide a versioned diverse template catalog
The system SHALL provide at least six normalized-text-unique templates and at least three style tags for every supported `(motif_type, query_intent)` pair.

#### Scenario: Template catalog validation
- **WHEN** the template catalog is loaded
- **THEN** it rejects duplicate template IDs, duplicate normalized text within a pair, missing required slots, unsupported pairs, and pairs below the diversity minimum

#### Scenario: Deterministic generation
- **WHEN** the same motif, template catalog version, generation seed, and requested query intent are verbalized repeatedly
- **THEN** the same template ID and query text are produced

#### Scenario: Wording diversity
- **WHEN** all templates for one supported motif/query-intent pair are rendered
- **THEN** they produce at least six distinct query strings spanning at least three style tags

### Requirement: Keep M1/M2 artifacts outside the experiment workflow
The system SHALL expose canonicalization, graph construction, motif extraction, and template verbalization as domain library APIs only in this change.

#### Scenario: Existing evidence experiment configuration
- **WHEN** maintained Hydra configs and dataset registries are inspected after this change
- **THEN** ISETrace is still absent from experiment dataset choices and existing evidence methods behave unchanged

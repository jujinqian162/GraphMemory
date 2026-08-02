## ADDED Requirements

### Requirement: Candidate sets contain exact-span provenance content units
For each query, the ISETrace adapter SHALL rank the same argument-content and output-content candidates derived from the referenced query-independent `ProvenanceGraph`, SHALL preserve their provenance node IDs and exact source spans, and MUST NOT expose labels, motif IDs, query intent, query origin, split assignment, or review metadata in candidate text or metadata.

#### Scenario: One graph serves multiple queries
- **WHEN** multiple natural or template query records reference the same graph ID
- **THEN** they receive identical ordered candidate IDs and source spans
- **AND** the persisted provenance graph fingerprint is identical

#### Scenario: Exact span has no candidate overlap
- **WHEN** a natural gold span overlaps no argument-content or output-content candidate in the referenced graph
- **THEN** preparation excludes and reports the uncompilable natural record before split allocation

## MODIFIED Requirements

### Requirement: ISETrace benchmark inputs are revision-pinned and content-addressed
The workflow SHALL identify the natural-query corpus and trajectory source independently, SHALL infer the pinned ISETrace revision from repository data registration, and SHALL persist the natural source digest, trajectory source identity, resolved split identity, and pinned revision in preparation provenance. Preparation MUST receive a different scientific cache identity when either source or the resolved trajectory-grouped allocation changes.

#### Scenario: Natural query corpus changes without trajectory changes
- **WHEN** the natural-query source content changes while the trajectory source is unchanged
- **THEN** the prepared artifact receives a different scientific cache identity
- **AND** the trajectory-grouped split is resolved as a new artifact

#### Scenario: Trajectory source identity changes without query changes
- **WHEN** the registered trajectory source revision or content identity changes while the natural-query source is unchanged
- **THEN** preparation fails on a conflicting authoring identity or produces a different scientific cache identity

### Requirement: Non-training workflow supports the derived natural test split
BM25, Dense, GraphRAG, and `provenance_path` SHALL remain non-training methods for ISETrace. When composed with the corpus-based ISETrace configuration, they SHALL retrieve and evaluate only the derived natural test split and MUST NOT schedule pair construction, frozen training encodings, model training, or checkpoint selection.

#### Scenario: BM25 uses the corpus-based ISETrace config
- **WHEN** BM25 is composed with an ISETrace corpus that resolves train, dev, and test query allocations
- **THEN** the workflow retrieves and evaluates only natural test records
- **AND** it schedules no pair-building or model-training task

#### Scenario: Provenance path remains training-free
- **WHEN** `provenance_path` is composed with the same ISETrace corpus
- **THEN** it receives the referenced query-independent graph for natural test requests
- **AND** no train or dev query is passed to its retrieval lifecycle

## REMOVED Requirements

### Requirement: Candidate sets contain native ToolOutput nodes only
**Reason**: The maintained ISETrace benchmark now evaluates exact source spans over chunked argument-content and output-content candidates. ToolOutput containers are graph connectors, not ranking candidates, and output-level gold IDs no longer match the natural v7 label contract.

**Migration**: Use the provenance content candidates and their exact source spans produced from `ProvenanceGraph`. Do not project content labels back to ToolOutput IDs.

### Requirement: Review admission is explicit
**Reason**: Per-record `allow_unreviewed` and `accepted_only` switches belong to the retired test-only benchmark shape. The natural source is now an immutable corpus input: generated data may be used for engineering runs, while formal claims require a separately reviewed and frozen natural source whose digest is recorded by the run.

**Migration**: Remove `allow_unreviewed` and `accepted_only` from ISETrace experiment configuration. Select and freeze the reviewed natural-query source before a formal run; changing that source creates a new preparation identity.

### Requirement: Evidence target policy is explicit
**Reason**: The `answer_only`, `support`, and `intent_aware` policies selected output-level labels from the retired motif-query benchmark. Natural v7 records now compile directly to exact source spans, and training-only templates use focused output-content labels.

**Migration**: Remove the evidence-target policy field. Use compiled exact spans for natural queries and focused output-content candidates for template supervision.

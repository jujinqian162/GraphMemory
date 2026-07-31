## ADDED Requirements

### Requirement: ISETrace benchmark inputs are revision-pinned and content-addressed
The workflow SHALL identify the query JSONL and trajectory JSONL independently, SHALL persist both digests and the pinned ISETrace revision in preparation provenance, and MUST invalidate preparation when either file changes.

#### Scenario: Query text changes without trajectory changes
- **WHEN** the query JSONL content changes while the trajectory file is unchanged
- **THEN** the prepared artifact receives a different scientific cache identity

#### Scenario: Trajectory changes without query changes
- **WHEN** the trajectory JSONL content changes while the query file is unchanged
- **THEN** the prepared artifact receives a different scientific cache identity

### Requirement: Candidate sets contain native ToolOutput nodes only
For each query, the adapter SHALL rank every `execution.tool_output` in the referenced trajectory, SHALL preserve provenance output IDs, and MUST NOT expose labels, motif IDs, query intent, or support metadata in candidate text or metadata.

#### Scenario: One graph serves multiple queries
- **WHEN** multiple query records reference the same graph ID
- **THEN** they receive identical ordered candidate IDs and the persisted provenance graph fingerprint is identical

#### Scenario: Label references an unknown output
- **WHEN** an answer, support, or dependency endpoint is absent from the referenced graph's ToolOutput nodes
- **THEN** preparation fails before retrieval

### Requirement: Physical graph topology is query-independent
The persisted `ProvenanceGraph` SHALL be built from the canonical trajectory before query projection and MUST contain no query text, answer, support IDs, motif metadata, or review metadata.

#### Scenario: Different questions over one trajectory
- **WHEN** two queries over one trajectory have different labels and wording
- **THEN** their graph ID, graph serialization, and graph fingerprint remain identical

### Requirement: Review admission is explicit
The dataset config SHALL declare `allow_unreviewed` or `accepted_only`. `allow_unreviewed` MAY admit unreviewed records for pilot execution but MUST reject records marked rejected. `accepted_only` SHALL admit only accepted or edited records.

#### Scenario: Formal policy receives unreviewed input
- **WHEN** `accepted_only` is configured and a query record remains unreviewed
- **THEN** that record is excluded and counts report the exclusion

### Requirement: Evidence target policy is explicit
The dataset config SHALL declare `answer_only`, `support`, or `intent_aware`. Under `intent_aware`, complete-chain and contributing-source queries SHALL use support outputs, while directional queries SHALL use answer outputs. Gold dependency edges MUST have both endpoints in the selected gold set.

#### Scenario: Directional query under intent-aware policy
- **WHEN** an upstream-source query has one answer output and two motif support outputs
- **THEN** only the answer output is gold evidence and no dependency with an excluded endpoint is retained

#### Scenario: Complete-chain query under intent-aware policy
- **WHEN** a complete-chain query has three support outputs and two logical dependencies
- **THEN** all three outputs and both dependencies are retained as gold

### Requirement: Non-training workflow supports test-only datasets
A non-training method SHALL be runnable with only a configured test split. Trainable methods MUST fail configuration validation unless train, dev, and test inputs are all present.

#### Scenario: BM25 uses test-only ISETrace config
- **WHEN** BM25 is composed with the ISETrace pilot dataset
- **THEN** the workflow prepares only test data and schedules no pair-building or model-training task

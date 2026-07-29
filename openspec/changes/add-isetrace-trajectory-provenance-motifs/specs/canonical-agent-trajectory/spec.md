## ADDED Requirements

### Requirement: Normalize ISETrace into a canonical ordered trajectory
The system SHALL convert each valid ISETrace session into a dataset-neutral `CanonicalTrajectory` containing source identity, embedded source intents, tool definitions, ordered canonical events, final response, and source metadata.

#### Scenario: Valid multi-intent trajectory
- **WHEN** an ISETrace record contains multiple embedded source intents and valid OpenAI-format messages
- **THEN** the adapter preserves every intent and emits stable ordered message, tool-call, and tool-output events
- **AND** no single synthetic Task node or merged task text is invented

#### Scenario: Multiple calls in one assistant message
- **WHEN** one assistant message contains multiple tool calls
- **THEN** each call becomes a distinct ordered `ToolCallEvent` with a stable event ID and source position

### Requirement: Preserve exact tool-call lineage
The canonical trajectory SHALL pair calls and outputs by source `tool_call_id` and validate execution invariants before publication.

#### Scenario: Complete call/output pairing
- **WHEN** every tool call has exactly one later output with the same call ID and tool name
- **THEN** the canonical trajectory is accepted and exposes the paired events through stable IDs

#### Scenario: Malformed pairing
- **WHEN** a call has no output, an output is orphaned or duplicated, or call/output tool names disagree
- **THEN** canonicalization fails with a structured record-specific reason

### Requirement: Preserve source semantics without overclaiming outcome truth
The canonical trajectory SHALL preserve the ISETrace success flag as `source_reported_success` and SHALL NOT treat it as an authoritative inferred outcome.

#### Scenario: Reported success contains error text
- **WHEN** a tool output has `success=true` but its text contains a traceback or structured error marker
- **THEN** the canonical event preserves `source_reported_success=true`
- **AND** compact ingestion counters may record the conflict without rewriting the event as a gold failure label

### Requirement: Retain future annotation anchors
Canonical message and tool-output events SHALL expose stable event IDs and text-bearing source positions suitable for later source-span semantic annotation.

#### Scenario: Later claim annotation
- **WHEN** a later NLP component identifies a claim in an assistant message
- **THEN** it can reference the canonical event ID and character offsets without modifying or reparsing an untyped raw trajectory object

### Requirement: Stream raw ISETrace records with compact quality accounting
The ISETrace adapter SHALL parse JSONL incrementally and expose compact accepted/rejected and call/output quality counters.

#### Scenario: Large raw shard
- **WHEN** the adapter consumes a multi-gigabyte trajectory shard
- **THEN** it yields canonical trajectories incrementally rather than loading the entire shard into memory

#### Scenario: Invalid record in non-strict iteration
- **WHEN** non-strict iteration encounters an invalid raw or canonical record
- **THEN** it increments a structured rejection reason and continues with later records

## ADDED Requirements

### Requirement: Temporal memory requests carry typed recency
The system SHALL represent Memory Stream recency inputs as typed request data on `TemporalMemoryRankingRequest`.

#### Scenario: Position recency request
- **WHEN** a dataset projector creates a temporal memory request for legacy position recency
- **THEN** it SHALL provide a typed position recency spec containing the candidate position map.

#### Scenario: Real-time recency request
- **WHEN** a dataset projector creates a temporal memory request for real-time recency
- **THEN** it SHALL provide a typed real-time recency spec containing the question datetime and candidate datetime map.

### Requirement: Memory Stream scoring consumes typed recency
Memory Stream scoring SHALL compute recency from the typed temporal recency spec instead of scorer-required metadata keys.

#### Scenario: Unsupported recency input
- **WHEN** a temporal memory request lacks a supported recency spec required by positive recency weight
- **THEN** Memory Stream scoring SHALL fail with a contract validation error before ranking silently proceeds.

#### Scenario: Metadata remains non-authoritative
- **WHEN** a temporal memory request contains metadata
- **THEN** Memory Stream scoring SHALL NOT require metadata keys such as `recency_mode`, `question_datetime`, or `datetime_by_item_id` for core recency computation.

### Requirement: Request-owned importance validation is shared
The system SHALL use one shared implementation to validate and normalize request-owned Memory Stream importance scores.

#### Scenario: Formal retrieval validates request importance
- **WHEN** Memory Stream retrieval runs without an external importance artifact
- **THEN** it SHALL validate missing, extra, non-numeric, and non-finite request-owned importance scores through the shared helper.

#### Scenario: Tuning validates request importance
- **WHEN** Memory Stream tuning runs without an external importance artifact
- **THEN** it SHALL validate request-owned importance scores through the same helper used by formal retrieval.

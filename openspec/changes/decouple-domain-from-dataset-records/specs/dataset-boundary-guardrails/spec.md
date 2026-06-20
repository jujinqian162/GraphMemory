## ADDED Requirements

### Requirement: Dataset-specific imports are boundary-limited
The system SHALL restrict dataset-specific record and projector imports to dataset packages, scripts, application use cases, stages, and explicit dataset adapters.

#### Scenario: Reusable packages do not import HotpotQA records
- **WHEN** production imports under reusable domain/model packages are scanned
- **THEN** packages such as `graphs`, `retrieval.execution`, `training_pairs`, `models`, `evaluation`, and generic `validation` do not import `graph_memory.datasets.hotpotqa.records`

#### Scenario: Reusable packages do not call HotpotQA projectors
- **WHEN** reusable domain/model packages are scanned
- **THEN** they do not import or instantiate `HotpotQAToTextRankingRequest`, `HotpotQAToGraphBuildRequest`, `HotpotQAToGraphRankingRequest`, `HotpotQAToTemporalMemoryRankingRequest`, or `HotpotQAToEvidenceEvaluationRequest`

### Requirement: Dataset field names do not re-enter domain internals
The system SHALL reject HotpotQA-specific field names in reusable domain/model internals except where they appear in explicitly dataset-owned modules or tests for dataset validation.

#### Scenario: Field-name guard scans reusable packages
- **WHEN** architecture tests scan reusable production packages
- **THEN** they fail on direct use of `candidate_sentences`, `gold_evidence_sentence_ids`, `sentence_id`, `sentence_index`, `document_sentence`, `source`, or `position` when those names are used as required domain contract fields

#### Scenario: Graph node guard scans contracts
- **WHEN** graph contract files are inspected
- **THEN** no generic graph node type exposes HotpotQA/document-sentence fields as required top-level fields

### Requirement: Dataset-aware boundaries are explicit
The system SHALL document and test the allowed places where HotpotQA records are loaded, validated, and projected.

#### Scenario: Stage/application boundary performs projection
- **WHEN** a HotpotQA workflow stage calls reusable domain/model code
- **THEN** it first projects HotpotQA records into consumer requests or domain task specs

#### Scenario: Dataset validators remain dataset-owned
- **WHEN** HotpotQA prepared artifacts are validated
- **THEN** HotpotQA-specific field checks remain in HotpotQA dataset validation code or clearly named HotpotQA validators

### Requirement: Durable docs describe the dataset/domain boundary
The system SHALL update durable documentation to distinguish dataset records, dataset projectors, consumer requests, domain artifacts, and supervised task specs.

#### Scenario: Contracts docs separate HotpotQA artifacts from domain artifacts
- **WHEN** maintainers read the contract docs
- **THEN** they can identify which types are HotpotQA-owned and which types are reusable domain contracts

#### Scenario: Architecture docs list forbidden dependency directions
- **WHEN** maintainers read architecture or handoff docs
- **THEN** they can see that reusable domain/model packages must not depend on dataset records or projectors

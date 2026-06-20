## ADDED Requirements

### Requirement: Graph artifacts use dataset-neutral node semantics
The system SHALL represent non-query nodes in `MemoryGraph` with graph-domain fields rather than HotpotQA/document-sentence fields.

#### Scenario: Graph node contract does not require sentence fields
- **WHEN** the graph contract for non-query nodes is inspected
- **THEN** it does not require `document_sentence`, `source`, `sentence_id`, or `position` as top-level fields

#### Scenario: Graph builder maps request nodes without dataset assumptions
- **WHEN** `GraphBuilder` builds a graph from `GraphBuildRequest`
- **THEN** every non-query graph node is derived from `GraphBuildNode` domain fields such as item id, kind, text, source reference, grouping key, sequence index, and metadata

### Requirement: Reusable domain packages consume projected requests
The system SHALL ensure reusable domain and model packages consume consumer-owned requests or dataset-neutral task specs after dataset projection.

#### Scenario: Retrieval execution uses request-level inputs
- **WHEN** retrieval execution runs a built retrieval method
- **THEN** it receives execution-ready ranking tasks or request objects and does not import HotpotQA records or HotpotQA projectors

#### Scenario: Ranked results are assembled from domain inputs
- **WHEN** ranked results are assembled
- **THEN** task id, candidate ids, retrieved subgraph, latency, and token accounting are derived from request/domain inputs rather than HotpotQA `candidate_sentences`

### Requirement: Training and model inputs are dataset-neutral
The system SHALL feed training pair construction, Dense-FT data construction, and R-GCN batching/training/inference with dataset-neutral supervised ranking inputs.

#### Scenario: Training pair construction does not read HotpotQA labels
- **WHEN** train pairs are built
- **THEN** positives and negatives are computed from domain candidate ids, evidence labels, text ranking requests, and graphs rather than `HotpotQALabelRecord.gold_evidence_sentence_ids`

#### Scenario: R-GCN inference does not reverse-project HotpotQA records
- **WHEN** checkpoint-backed graph retriever inference ranks a `GraphRankingRequest`
- **THEN** it batches directly from request/domain inputs and does not construct a `HotpotQARankingRecord`

#### Scenario: Dense-FT data uses text requests and evidence labels
- **WHEN** Dense-FT examples or IR evaluator payloads are built
- **THEN** they consume `TextRankingRequest`, `EvidenceLabel`, and `TrainPairRecord` inputs instead of HotpotQA ranking and label records

### Requirement: Evaluation consumes evaluation requests
The system SHALL keep evaluation and failure-case generation behind `EvidenceEvaluationRequest` / `EvidenceLabel` or equivalent dataset-neutral evaluation contracts.

#### Scenario: Evaluation service has no HotpotQA compatibility path
- **WHEN** evaluation metrics are computed
- **THEN** the evaluation service consumes an `EvidenceEvaluationRequest` and does not call HotpotQA projectors

#### Scenario: Failure cases use evidence labels
- **WHEN** failure cases are generated
- **THEN** gold evidence ids are read from `EvidenceLabel.gold_evidence_item_ids` rather than HotpotQA label fields

### Requirement: Validators receive domain expectations
The system SHALL make reusable validators validate against explicit domain expectations rather than HotpotQA ranking records.

#### Scenario: Graph validation uses expected graph item ids
- **WHEN** graph artifacts are validated
- **THEN** the validator receives expected graph item ids or graph build requests and does not derive them from `candidate_sentences`

#### Scenario: Ranked-result validation uses expected candidate ids
- **WHEN** ranked results are validated
- **THEN** the validator receives expected candidate ids or text ranking requests and does not derive them from HotpotQA candidate sentences

#### Scenario: Train-pair validation uses evidence labels
- **WHEN** train pairs are validated
- **THEN** the validator receives evidence labels and expected candidate ids rather than HotpotQA label records

### Requirement: Memory Stream importance uses temporal/domain inputs
The system SHALL make Memory Stream importance digesting, selection, and validation operate on `TemporalMemoryRankingRequest` or an equivalent dataset-neutral importance task spec.

#### Scenario: Importance digest ignores dataset record shape
- **WHEN** an importance content digest is computed
- **THEN** the digest is based on item ids, item text, source reference, ordering metadata, and task id from domain inputs rather than HotpotQA candidate sentence records

#### Scenario: Importance validation uses temporal request candidate ids
- **WHEN** importance score mappings are validated
- **THEN** expected item ids come from temporal/domain inputs rather than `HotpotQARankingRecord`

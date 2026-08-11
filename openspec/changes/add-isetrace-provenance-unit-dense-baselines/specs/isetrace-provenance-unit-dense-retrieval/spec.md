## Purpose

Defines matched graph-free Dense controls over ISETrace provenance content units so candidate segmentation can be measured independently from provenance traversal and graph convolution.

## ADDED Requirements

### Requirement: Dense methods expose an explicit candidate-view variant
The existing `dense` and `dense_ft` methods SHALL accept `variant=flat|provenance_unit`, SHALL default to `flat`, and SHALL reject `provenance_unit` outside ISETrace before preparation or retrieval. The resolved variant MUST participate in scientific configuration, cache identity, artifact provenance, and final run output.

#### Scenario: Preserve the existing default
- **WHEN** Dense or Dense-FT is composed without a variant override
- **THEN** the resolved variant is `flat`
- **AND** existing candidate, training, and retrieval behavior is unchanged

#### Scenario: Reject an evidence dataset
- **WHEN** either method is composed with `variant=provenance_unit` on HotpotQA, 2Wiki, or MuSiQue
- **THEN** configuration resolution fails before preparation or retrieval

### Requirement: Provenance-unit variants use the native candidate view without graph input
Both provenance-unit variants SHALL rank the exact `provenance_candidates` persisted in each prepared ISETrace ranking record through graph-free text requests. They MUST NOT receive a `ProvenanceGraph`, graph edge, graph-derived feature, graph identifier feature, motif, query-type label, or native provenance trace.

#### Scenario: Run frozen provenance-unit Dense
- **WHEN** `dense variant=provenance_unit` is run on a prepared ISETrace task
- **THEN** it receives the query text and all provenance candidates with their exact source spans
- **AND** it emits a complete Dense ranking whose run result records `variant=provenance_unit`
- **AND** no graph artifact is loaded or supplied to the retriever

### Requirement: Frozen provenance-unit Dense isolates path traversal
`dense variant=provenance_unit` SHALL reuse the same frozen encoder settings, query/passage formatting, cosine scoring, batching, and deterministic tie-breaking as the Dense initialization inside `provenance_path`.

#### Scenario: Compare against provenance-path initialization
- **WHEN** provenance-unit Dense and the Dense initialization of `provenance_path` receive the same ISETrace task, encoder, and candidates
- **THEN** their pre-traversal candidate scores and order are identical

### Requirement: Exact source spans define provenance-unit Dense-FT supervision
The system SHALL map every ISETrace training or development gold source span to every provenance candidate whose source spans overlap it. All positives MUST belong to the task-local provenance candidate set, and pair or model preparation MUST fail with the task ID when a task has no overlapping provenance candidate.

#### Scenario: Map one gold span to provenance units
- **WHEN** a gold source span overlaps one or more provenance content candidates
- **THEN** every overlapping candidate is included in the task-local positive set
- **AND** no flat chunk ID or graph relation is included in the label

#### Scenario: Reject unmappable supervision
- **WHEN** a training or development task has gold spans but no overlapping provenance candidate
- **THEN** preparation fails before optimization and identifies the task

### Requirement: Provenance-unit Dense-FT is training-matched to flat Dense-FT
`dense_ft variant=provenance_unit` SHALL reuse the existing Dense-FT trainer, encoder, loss, optimizer, epoch count, batching policy, task-local development evaluator, and configured random seed. Its effective ISETrace negative sampling SHALL remain graph-free with `hard_graph_neighbor_per_positive=0`; all other configured sampling values SHALL match flat ISETrace Dense-FT.

#### Scenario: Build matched training pairs
- **WHEN** flat and provenance-unit Dense-FT are composed with the same ISETrace profile and seed
- **THEN** their effective negative-sampling configurations differ only in the candidate IDs and texts induced by the selected variant
- **AND** both have graph-neighbor sampling disabled
- **AND** each pair artifact records its resolved variant

#### Scenario: Select a checkpoint
- **WHEN** provenance-unit Dense-FT completes an epoch on ISETrace
- **THEN** development queries are ranked only against their own trajectory's provenance candidates
- **AND** the checkpoint is selected by development Recall@5 under the same rule as flat ISETrace Dense-FT

### Requirement: Flat and provenance-unit Dense-FT checkpoints are not interchangeable
Dense-FT model metadata SHALL record `variant=flat|provenance_unit`. Retrieval MUST reject a checkpoint whose recorded variant does not equal the requested variant, without translating or guessing provenance-unit identity. Existing metadata without an explicit variant SHALL retain its historical flat meaning.

#### Scenario: Load the matching checkpoint
- **WHEN** Dense-FT with `variant=provenance_unit` receives a checkpoint recorded for that variant
- **THEN** the existing SentenceTransformer model directory is loaded and used to rank provenance candidates

#### Scenario: Reject a flat checkpoint
- **WHEN** Dense-FT with `variant=provenance_unit` receives a checkpoint recorded as flat or without the new field
- **THEN** retrieval fails before ranking with an explicit variant mismatch

### Requirement: Provenance-unit controls use the existing exact-span evaluation and output contract
Both provenance-unit variants SHALL preserve candidate source spans through a complete ranking and SHALL use the same ISETrace test split, fixed token-budget evaluation, per-query output, run delivery, and aggregation contracts as the flat variants. Path and edge accuracy MUST remain unavailable because neither variant predicts labeled provenance structure.

#### Scenario: Complete frozen evaluation
- **WHEN** provenance-unit Dense finishes ISETrace test retrieval
- **THEN** Recall, MRR, token-budget Coverage, Full Support, span F1, and evidence density are produced under method `dense` and variant `provenance_unit`
- **AND** no train, development, pair, model, graph, or frozen-R-GCN stage is scheduled

#### Scenario: Complete fine-tuned evaluation
- **WHEN** provenance-unit Dense-FT finishes an ISETrace run
- **THEN** pair, model-directory, training-metric, prediction, exact-span metric, aggregate, and reproducibility artifacts record method `dense_ft` and variant `provenance_unit`
- **AND** no EvidenceGraph, ProvenanceGraph retrieval input, or frozen-R-GCN embedding stage is scheduled

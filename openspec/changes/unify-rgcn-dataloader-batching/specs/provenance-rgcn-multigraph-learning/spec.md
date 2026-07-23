## ADDED Requirements

### Requirement: Provenance batches preserve task ownership for every scored object
The system SHALL represent a provenance graph batch with one query per task, task-partitioned candidates, and task-partitioned logical transitions. Every model-facing candidate and transition index SHALL be offset into the disconnected union while retaining enough offsets and metadata to recover its owning task.

#### Scenario: Candidate query ownership survives collation
- **WHEN** candidates from multiple provenance tasks are collated
- **THEN** every candidate is scored against the query node belonging to its own task and no scalar query is broadcast across tasks

#### Scenario: Logical transition ownership survives collation
- **WHEN** logical transitions from multiple tasks are collated
- **THEN** every transition source, target, and query index lies in its owning task interval and transition offsets partition edge logits by task

#### Scenario: Empty transitions are supported
- **WHEN** one or more collated tasks have no logical transitions
- **THEN** their transition offset slice is empty, candidate scoring still succeeds, and other tasks' transitions remain correctly aligned

#### Scenario: Candidate and transition outputs can be split
- **WHEN** a batched provenance forward returns flat candidate and edge logits
- **THEN** candidate and transition offsets recover exactly the original per-task order, IDs, and logical transition metadata

### Requirement: Provenance R-GCN performs true multi-graph model execution
The provenance R-GCN SHALL execute node projection, relational message passing, candidate scoring, and logical-edge scoring once over the disconnected-union batch. A batch-size-one request SHALL use the same forward contract.

#### Scenario: Batched node states are isolated
- **WHEN** multiple provenance graphs with no cross-task edges are forwarded with dropout disabled
- **THEN** each task's node states equal the corresponding single-task forward within declared floating-point tolerance

#### Scenario: Batched logits match single-task logits
- **WHEN** the same model evaluates multiple provenance tasks separately and as one disconnected union with dropout disabled
- **THEN** candidate and logical-edge logits split from the batched output match the separate outputs within declared floating-point tolerance

#### Scenario: Training uses one optimizer step per graph batch
- **WHEN** a provenance training DataLoader emits a batch containing `B` task graphs
- **THEN** the model receives one disconnected-union forward, one backward pass, and one optimizer step rather than `B` sequential single-graph forwards

### Requirement: Candidate-loss protocol remains provenance-candidate-loss-v2
The batching refactor SHALL retain `candidate_loss_protocol=provenance-candidate-loss-v2` and `candidate_loss_type=task_balanced_pairwise_logistic`. For each task, every materialized positive SHALL be compared with every unique materialized negative, comparisons SHALL be averaged within that task, and task candidate losses SHALL NOT be combined by a global comparison mean.

#### Scenario: Pairwise comparisons remain task-local
- **WHEN** a graph batch contains two or more tasks with positive and negative candidate targets
- **THEN** the comparison count equals `sum_t(|P_t|*|N_t|)` and contains no positive-negative pair drawn from different tasks

#### Scenario: Unpaired candidates remain excluded from candidate loss
- **WHEN** a candidate participates in graph propagation but has no materialized train-pair target
- **THEN** it remains available for scoring/inference but contributes neither a positive nor a negative candidate-loss comparison

#### Scenario: Task candidate losses are equally weighted
- **WHEN** tasks in one graph batch have different numbers of selected negatives or pairwise comparisons
- **THEN** each task contributes one equally weighted candidate-loss scalar after its own comparison mean

#### Scenario: Task without selected negatives is retained
- **WHEN** a valid task has positive supervision but no selected negative
- **THEN** its candidate-loss term is differentiable zero and its logical-edge term and task weight remain present

### Requirement: Logical-edge supervision remains task-local class-balanced BCE
The system SHALL preserve the existing logical-edge loss: edge targets align with each task's logical transitions, positive weighting is calculated from that task's positive and negative transitions, and edge loss is averaged within task before task-level batch reduction.

#### Scenario: Edge positive weighting is not global
- **WHEN** two tasks in the same graph batch have different positive/negative logical-edge ratios
- **THEN** each task uses its own class-balance weight rather than a weight computed from all batch transitions

#### Scenario: No-transition task is supported
- **WHEN** a task has no logical transitions
- **THEN** its logical-edge loss is differentiable zero and does not consume another task's edge logits or targets

#### Scenario: Candidate and edge objectives retain configured weights
- **WHEN** task total loss is computed
- **THEN** it equals configured candidate-loss weight times that task's v2 candidate loss plus configured edge-loss weight times that task's logical-edge BCE

### Requirement: Provenance graph batches are reduced by actual task count
The provenance trainer SHALL reduce task total losses as an equal-weight mean over the actual tasks in each DataLoader batch. It MUST NOT normalize with the nominal configured graph-batch size when the final batch is incomplete, and it MUST NOT retain gradients across DataLoader batches.

#### Scenario: Short final graph batch is normalized exactly
- **WHEN** the final DataLoader batch contains `m < per_device_graph_batch_size` tasks
- **THEN** its gradient equals the mean of those `m` task losses rather than their sum divided by the configured graph batch size

#### Scenario: Gradient clipping occurs after batch normalization
- **WHEN** a DataLoader batch completes its backward pass
- **THEN** its actual-task mean is clipped and applied in that batch's optimizer step before the next batch begins

### Requirement: Batched dev execution retains task-local structured inference
The provenance dev path SHALL use ordered graph DataLoader batches for model execution and SHALL split outputs before candidate sorting, structured promotion, edge abstention, trace construction, and metric evaluation. Public single-task retrieval SHALL execute the same batch contract with one graph.

#### Scenario: Structured reranking cannot cross tasks
- **WHEN** a dev batch contains multiple tasks
- **THEN** candidate pools, seed sources, transition conflicts, protected prefixes, top-k filtering, and abstention decisions are evaluated independently within each task

#### Scenario: Batched dev metrics match single-task reference
- **WHEN** dropout is disabled and identical dev tasks are evaluated separately and through batched model execution
- **THEN** per-task rankings, logical edges, traces, and aggregate dev metrics match within declared tolerance

#### Scenario: Public rank_task remains compatible
- **WHEN** `ExecutionProvenanceRgcnRetriever.rank_task` receives one valid provenance request
- **THEN** it returns the existing task-local retrieval result shape while internally using the batch-size-one tensor/model/output-splitting path

### Requirement: Provenance checkpoint identity describes batch semantics without changing loss v2
The provenance checkpoint schema SHALL advance to a version that records the v2 candidate-loss protocol and the new disconnected-union batch semantics separately. The strict loader MUST NOT silently reinterpret a legacy `batch_size` as a graph batch count.

#### Scenario: New checkpoint records scientific identities
- **WHEN** a provenance model is saved after this change
- **THEN** its payload records checkpoint schema version, `provenance-candidate-loss-v2`, pairwise loss type, disconnected-union batch semantics, and per-device graph batch

#### Scenario: Candidate-loss protocol tag is unchanged
- **WHEN** only tensor collation, DataLoader iteration, or physical graph batching changes
- **THEN** the checkpoint still declares `candidate_loss_protocol=provenance-candidate-loss-v2` and does not declare v3 or another loss protocol

#### Scenario: Legacy checkpoint is not silently migrated
- **WHEN** the strict new loader receives a provenance checkpoint whose training config contains only the old ambiguous `batch_size`
- **THEN** it rejects the checkpoint with an explicit schema/semantics error rather than guessing the graph-batch value

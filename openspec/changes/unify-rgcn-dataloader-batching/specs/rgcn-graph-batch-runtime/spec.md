## ADDED Requirements

### Requirement: R-GCN physical batch semantics are explicit and uniform
The system SHALL define `per_device_graph_batch_size` for both evidence and provenance R-GCN as the number of task graphs collated into one device-local model forward, backward pass, and optimizer step. The trainers MUST NOT retain gradients across DataLoader batches or use one `batch_size` field with different meanings.

#### Scenario: Evidence and provenance consume the same batch definition
- **WHEN** either R-GCN trainer resolves its training configuration
- **THEN** `per_device_graph_batch_size=N` means that each full DataLoader batch contains `N` task graphs and produces exactly one optimizer/global step

#### Scenario: Final batch is not misreported
- **WHEN** the final DataLoader batch contains fewer tasks than the configured graph batch
- **THEN** the optimizer update and metrics use the actual observed task or supervision count

### Requirement: Shared graph collation produces an isolated disconnected union
The system SHALL provide one shared graph collation contract used by both R-GCN families. Collation SHALL concatenate node tensors, offset all edge/query indices, preserve task order and identity, and create a disconnected union with no synthetic cross-task edge.

#### Scenario: Multiple task graphs are collated
- **WHEN** task graphs with node counts `n_0 ... n_(B-1)` are collated
- **THEN** the result contains `B` query indices, `B` task identities, and offsets `[0, n_0, n_0+n_1, ..., sum(n)]`

#### Scenario: Edge endpoints remain task-local
- **WHEN** a graph edge from task `t` is offset into the disconnected union
- **THEN** both of its endpoints lie within task `t`'s node-offset interval and no edge connects two task intervals

#### Scenario: One-task collation uses the same contract
- **WHEN** exactly one task graph with `N` nodes is collated
- **THEN** the result has task offsets `[0, N]` and is accepted by the same model and device-transfer path as a multi-task batch

#### Scenario: Device transfer preserves graph metadata
- **WHEN** a shared graph batch is moved to a training device
- **THEN** every tensor field moves to that device while task IDs, node IDs, ordering, and offsets remain unchanged

### Requirement: Frozen features are materialized before DataLoader iteration
The system SHALL represent each evidence or provenance dataset item as one validated CPU task tensor with frozen encoder features already materialized. The encoder MUST NOT be invoked by DataLoader `__getitem__`, a DataLoader worker, or an epoch-time collation function.

#### Scenario: Multiple epochs reuse frozen features
- **WHEN** the same R-GCN dataset is iterated for more than one epoch
- **THEN** task order may change but the frozen encoder is not called again for already materialized task tensors

#### Scenario: DataLoader owns task sampling rather than fixed batch sampling
- **WHEN** a training DataLoader is created
- **THEN** its map-style dataset exposes individual task items and its custom collator constructs the current graph batch after task sampling

#### Scenario: Initial worker policy is safe
- **WHEN** either R-GCN DataLoader is created by this change
- **THEN** it uses `num_workers=0` and contains no CUDA encoder or CUDA task tensor in the dataset

### Requirement: Training and dev DataLoaders have deterministic complete iteration
Each R-GCN family SHALL use a PyTorch DataLoader with an explicit family-specific collator. Training SHALL use seeded task-level shuffling; dev evaluation SHALL preserve declared task order; both SHALL retain incomplete final batches.

#### Scenario: Seeded training order is reproducible
- **WHEN** two training loaders are created from identical task items and the same random seed
- **THEN** they produce the same task order and graph-batch boundaries for corresponding epochs

#### Scenario: Training reshuffles task graphs across epochs
- **WHEN** a multi-task training loader is iterated for successive epochs
- **THEN** the seeded sampler advances deterministically and is not constrained to the fixed preconstructed batch boundaries used before this change

#### Scenario: Dev order remains stable
- **WHEN** a dev loader is iterated
- **THEN** it emits every task exactly once in declared dataset order with shuffling disabled

#### Scenario: Incomplete batch is retained
- **WHEN** task count is not divisible by `per_device_graph_batch_size`
- **THEN** the final smaller batch is emitted and no task is dropped

### Requirement: Evidence R-GCN scientific behavior is preserved
The evidence R-GCN DataLoader migration SHALL preserve its existing graph construction, model computation, global training-set positive weighting, sample-level BCE reduction, dev prediction ordering, and checkpoint selection semantics.

#### Scenario: Evidence loss remains sample-weighted
- **WHEN** one evidence graph batch contains tasks with different numbers of supervised nodes
- **THEN** BCE is reduced over supervised samples rather than changed to an equal-weight mean of task losses

#### Scenario: Evidence single-batch parity is checked
- **WHEN** dropout is disabled and the same task items are processed through the previous collator reference and the new shared-plus-evidence collator
- **THEN** graph tensors, logits, labels, and loss agree within the declared floating-point tolerance

#### Scenario: Evidence task shuffle does not alter task content
- **WHEN** training shuffles evidence task items before collation
- **THEN** each task retains its original graph, frozen features, pairs, labels, node IDs, and query binding

### Requirement: Batch execution is observable and auditable
Each R-GCN training run SHALL expose enough counts to audit variable graph batches and loss denominators.

#### Scenario: Epoch training metrics are recorded
- **WHEN** an R-GCN epoch completes
- **THEN** its metrics include optimizer step count, task count, configured graph batch, actual tasks per optimizer step, and the model-specific loss denominator count

#### Scenario: Variable graph load is summarized
- **WHEN** a training or dev dataset is materialized and iterated
- **THEN** run diagnostics report task, node, edge, and applicable supervised-item/candidate/transition counts without uploading the underlying task tensors

### Requirement: Evidence checkpoint identity describes graph-batch semantics
The evidence R-GCN checkpoint schema SHALL advance from v2 to v3 and SHALL store explicit disconnected-union graph-batch semantics. The strict loader MUST NOT accept an old `batch_size` field as if it were the new configuration.

#### Scenario: New evidence checkpoint records explicit fields
- **WHEN** an evidence R-GCN checkpoint is saved after this change
- **THEN** its training configuration records the per-device graph batch under checkpoint schema v3

#### Scenario: Legacy evidence checkpoint is rejected
- **WHEN** the strict schema-v3 evidence loader receives a schema-v2 checkpoint with only the old `batch_size` field
- **THEN** it reports an explicit schema/semantics mismatch instead of guessing the new fields

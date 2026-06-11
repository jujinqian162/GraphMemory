## ADDED Requirements

### Requirement: Default graph features reuse one dense forward result
The system SHALL allow the default dense graph feature provider to produce ordered node embeddings and dense seed signals from the same normalized query and passage embedding result.

#### Scenario: Compatible default providers are shared
- **WHEN** trainable graph batching receives the same joint dense feature provider as both its embedding and seed dependency
- **THEN** each task group is encoded once and seed scores are derived from those returned embeddings without a second sentence-encoder forward pass

#### Scenario: Dense seed semantics remain equivalent
- **WHEN** seed signals are derived from the shared normalized embeddings
- **THEN** signal scores, ranks, rank percentiles, descending score order, node-ID tie-breaks, and complete memory-node coverage match the existing dense retriever-backed provider

#### Scenario: Independently injected providers are respected
- **WHEN** embedding and seed dependencies are different objects or do not expose the joint capability
- **THEN** the system uses each dependency through its existing or bulk-compatible contract and does not substitute dense scores for the injected seed semantics

### Requirement: Graph batch construction batches embedding work by task group
The system SHALL build graph node embeddings for all tasks in one task-graph batch through the provider bulk path while preserving graph tensor and metadata invariants.

#### Scenario: Training batch equivalence
- **WHEN** a training batch is built from the same tasks, graphs, pairs, model config, and deterministic providers
- **THEN** node embeddings, node features, edge tensors, relation IDs, edge weights, offsets, query indices, sample indices, labels, task IDs, node IDs, and sample order match the single-task construction behavior

#### Scenario: Full-ranking batch equivalence
- **WHEN** a dev or inference full-ranking batch is built through bulk graph features
- **THEN** all memory nodes remain present in the same task and node order with unchanged labels and ranking metadata

### Requirement: Frozen dev batches are reused within training
The system SHALL construct immutable frozen dev graph batches at most once per trainable graph training invocation and reuse them for every epoch's model evaluation.

#### Scenario: Multi-epoch dev evaluation
- **WHEN** training runs for multiple epochs over unchanged dev task inputs, graphs, labels, model config, and frozen providers
- **THEN** dev text encoding, seed feature construction, and graph tensorization occur once while model forward evaluation occurs once per epoch

#### Scenario: Reused batches are not mutated
- **WHEN** a cached CPU dev batch is moved to the training device for evaluation
- **THEN** device movement produces a separate batch value and leaves the reusable CPU batch unchanged

### Requirement: Frozen feature reuse is invocation-scoped
The system MUST keep frozen dense feature reuse bounded to the active graph batch or training invocation and MUST NOT introduce an implicit process-global or persistent embedding cache.

#### Scenario: Training invocation completes
- **WHEN** a trainable graph training invocation returns or fails
- **THEN** no new global cache contract retains its task embeddings or dev batches for later invocations

#### Scenario: Persistent artifacts remain unchanged
- **WHEN** training, retrieval, tuning, or pair-generation commands complete
- **THEN** they do not write a new embedding-cache artifact and existing output and checkpoint schemas remain unchanged

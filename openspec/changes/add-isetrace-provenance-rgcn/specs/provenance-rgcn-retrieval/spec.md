## ADDED Requirements

### Requirement: Training-only template supervision reuses provenance motifs
The system SHALL derive template query supervision from existing query-independent provenance motifs without mutating the graph or restoring legacy answer/support contracts.

#### Scenario: Generate a dependency template query
- **WHEN** an assigned train or dev trajectory contains an eligible dependency motif and authoring target
- **THEN** the renderer produces a deterministic query from safe graph descriptions
- **AND** the query contains no graph ID, event ID, node ID, binding hash, or hidden answer value

#### Scenario: Label focused output content
- **WHEN** a motif target identifies focus and participant output IDs
- **THEN** output-content candidates owned by focused outputs are positive template candidates
- **AND** participant outputs that are not focused are not positive merely because they belong to the motif

#### Scenario: Preserve the physical graph
- **WHEN** any number of natural or template queries are prepared for one trajectory
- **THEN** every query references the same persisted `ProvenanceGraph` fingerprint

### Requirement: Provenance R-GCN reuses the maintained graph-retriever runtime
The system SHALL implement `provenance_rgcn` as an execution-provenance adapter around the existing frozen-encoder R-GCN neural, batching, sampling, training, checkpoint, and selection components.

#### Scenario: Register the method
- **WHEN** Registry is inspected
- **THEN** `provenance_rgcn` supports the execution-provenance task family
- **AND** it does not support evidence-retrieval requests
- **AND** existing evidence R-GCN methods retain their current definitions

#### Scenario: Build a training batch
- **WHEN** provenance tasks are materialized for training
- **THEN** they use the shared `TaskGraphTensor` and disconnected-union graph batching contracts
- **AND** existing R-GCN message-passing and node-scoring modules execute the batch
- **AND** no second graph-convolution implementation is introduced

#### Scenario: Use current checkpoints only
- **WHEN** a trained provenance R-GCN checkpoint is saved and loaded
- **THEN** strict current model config, method identity, encoder identity, relation vocabulary, and model tensors round-trip
- **AND** no deleted provenance-checkpoint compatibility loader is used

### Requirement: Query conditioning does not mutate provenance graphs
The system SHALL condition candidate scoring on an ephemeral query representation while keeping the persisted graph query-independent.

#### Scenario: Tensorize one query
- **WHEN** a natural or template query over a trajectory is tensorized
- **THEN** one ephemeral disconnected query node is appended to the task tensor
- **AND** no query node or query-derived edge is added to the persisted `ProvenanceGraph`
- **AND** the scorer gathers the correct query state for every candidate in that task

#### Scenario: Batch multiple queries
- **WHEN** multiple task graphs are collated into one disconnected-union batch
- **THEN** each candidate is paired only with its owning query
- **AND** no provenance edge, query index, candidate index, or loss target crosses a task interval

### Requirement: Fixed physical-relation tensorization
The system SHALL tensorize the existing provenance graph with one fixed query-independent relation policy.

#### Scenario: Tensorize enabled edges
- **WHEN** a provenance graph contains returns, argument/content ownership, feeds, resource access, or content-order edges
- **THEN** the tensorizer emits explicit forward and reverse relation IDs with uniform weight 1.0

#### Scenario: Exclude temporal shortcut edges
- **WHEN** a graph contains `temporal.precedes`
- **THEN** those edges do not enter R-GCN message passing

#### Scenario: Keep forbidden metadata out of features
- **WHEN** node tensors are built
- **THEN** node IDs, message ordinals, motif types, template identities, query origins, and split assignments are not model features

### Requirement: Train with natural-span and template-focus labels
The system SHALL map natural exact spans and template focused outputs into candidate-level supervision and reuse the existing negative-sampling and node-ranking behavior.

#### Scenario: Map a natural label
- **WHEN** a natural query gold span overlaps a provenance argument or output content candidate
- **THEN** that candidate is a positive training item

#### Scenario: Map a template label
- **WHEN** a template target focuses one or more ToolOutputs
- **THEN** their owned output-content candidates are positive training items

#### Scenario: Sample negatives
- **WHEN** training pairs are built
- **THEN** configured easy-random, BM25-hard, Dense-hard, and graph-neighbor negatives are sampled through the maintained pair-sampling behavior
- **AND** nonfocused participant candidates remain eligible negatives

#### Scenario: Optimize node ranking only
- **WHEN** the provenance R-GCN is trained
- **THEN** its loss supervises candidate ranking
- **AND** no dependency-edge prediction head or auxiliary edge loss is created

### Requirement: Natural-first selection and natural-only evaluation
The system SHALL use natural dev behavior to select the model and the existing exact-span ISETrace suite to evaluate a natural-only test split.

#### Scenario: Evaluate mixed dev
- **WHEN** a training epoch completes
- **THEN** natural and template dev metrics are computed separately
- **AND** checkpoint selection uses the natural dev metrics as its primary input

#### Scenario: Retrieve formal test queries
- **WHEN** the trained method runs on test
- **THEN** every request is a natural query
- **AND** ranked candidates retain exact source spans
- **AND** the existing span recall/coverage, Full Support, span F1, MRR, evidence-density, and token-budget metrics are produced

#### Scenario: Avoid unsupported edge claims
- **WHEN** test labels contain no independently annotated dependency edges
- **THEN** path and edge accuracy remain unavailable rather than being derived from the input graph

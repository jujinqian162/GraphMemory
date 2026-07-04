## ADDED Requirements

### Requirement: Public learned graph R-GCN method
The system SHALL expose `learned_graph_rgcn_retriever` as a first-class retrieval method with its own registry definition, method config type, checkpoint method identity, and result-table method name.

#### Scenario: Method is listed and graph aware
- **WHEN** the method registry is queried for available retrieval methods
- **THEN** `learned_graph_rgcn_retriever` is present and supports graph path metrics

#### Scenario: Existing R-GCN behavior is unchanged
- **WHEN** `dense_rgcn_graph_retriever` or `dense_ft_rgcn_graph_retriever` is selected
- **THEN** the system uses their existing graph artifacts, configs, model behavior, and checkpoint identity

### Requirement: Method-local proposal graph stage
The system SHALL build proposal graph artifacts for `learned_graph_rgcn_retriever` through a method-local `proposal_graphs` dependency chain and MUST NOT require another method's shared `graphs` stage.

#### Scenario: Only learned method is selected
- **WHEN** an experiment is planned with only `learned_graph_rgcn_retriever`
- **THEN** the plan contains proposal graph stages for the learned method and does not contain unrelated shared graph stages for other methods

#### Scenario: Learned and existing graph methods are selected together
- **WHEN** an experiment is planned with both `dense_rgcn_graph_retriever` and `learned_graph_rgcn_retriever`
- **THEN** each method reads its own graph artifact path and the learned method's proposal graph stage is not downstream of the existing graph stage

### Requirement: Configurable rank edge sparse loss
The system SHALL train the learned graph R-GCN with configurable `rank_loss_weight`, `edge_loss_weight`, and `sparse_loss_weight` values from method config, defaulting to `1.0`, `0.2`, and `0.05`.

#### Scenario: Defaults are applied
- **WHEN** a learned graph R-GCN method config omits custom loss weights only where defaults are allowed
- **THEN** the training config uses rank, edge, and sparse weights of `1.0`, `0.2`, and `0.05`

#### Scenario: Weights affect total loss
- **WHEN** training computes learned graph R-GCN loss
- **THEN** total loss equals the weighted sum of rank BCE, edge BCE, and sparse gate penalty

#### Scenario: No edge labels are available
- **WHEN** a training batch has no labeled candidate edges
- **THEN** edge loss is zero on the correct device, `edge_loss_sample_count` is zero, and training continues

### Requirement: Learned edge gate scales proposal graph messages
The system SHALL learn an edge gate for candidate proposal edges and use the gate to scale R-GCN message edge weights for the learned method.

#### Scenario: Gate tensor aligns with graph edges
- **WHEN** a learned graph training batch is tensorized
- **THEN** edge gate features, gate logits, and edge weights align with the message edges consumed by R-GCN

#### Scenario: Closed gates suppress proposal messages
- **WHEN** learned edge gates are zero
- **THEN** proposal edge messages do not contribute to relation-aware message passing

### Requirement: Label-only edge supervision
The system SHALL use `gold_dependency_edges` only as label-side supervision for edge loss and path metrics, and MUST NOT insert them into proposal graph artifacts or input-visible graph tensors.

#### Scenario: Proposal graph excludes gold fields
- **WHEN** proposal graphs are built from 2Wiki records that include supporting facts, evidences, evidence IDs, answers, and gold dependency edges in label records
- **THEN** the proposal graph artifact contains only input-visible graph data and excludes those gold fields

#### Scenario: Edge labels come from matching candidate endpoints
- **WHEN** a proposal edge endpoint pair matches a `gold_dependency_edges` endpoint pair for the same task
- **THEN** that candidate edge is labeled positive for edge loss without adding any new graph edge

### Requirement: Checkpoint-backed inference reuse
The system SHALL load learned graph R-GCN checkpoints through the existing checkpoint-backed graph retrieval path and SHALL reject checkpoints whose method identity does not match the selected method.

#### Scenario: Learned checkpoint is loaded for learned retrieval
- **WHEN** `learned_graph_rgcn_retriever` retrieval loads a learned graph R-GCN checkpoint
- **THEN** predictions include ranked nodes, method provenance, and graph-aware `retrieved_subgraph` derived from the proposal graph

#### Scenario: Method mismatch is rejected
- **WHEN** a `dense_rgcn_graph_retriever` checkpoint is loaded for `learned_graph_rgcn_retriever`, or the reverse
- **THEN** loading fails before retrieval with a clear method mismatch error

### Requirement: HotpotQA workflow integration
The system SHALL allow HotpotQA experiment configs to select `learned_graph_rgcn_retriever` while preserving HotpotQA's lack of edge labels.

#### Scenario: HotpotQA plan includes proposal graphs
- **WHEN** HotpotQA evidence retrieval is planned with `learned_graph_rgcn_retriever`
- **THEN** the plan includes proposal graph, pairs, train, retrieve, evaluate, and aggregate stages for the learned method

#### Scenario: HotpotQA edge loss has no samples
- **WHEN** learned graph R-GCN training consumes HotpotQA labels
- **THEN** edge loss sample count is zero and node ranking training remains valid

### Requirement: 2Wiki workflow integration
The system SHALL allow 2Wiki experiment configs to select `learned_graph_rgcn_retriever` and use 2Wiki dependency labels only for edge loss labels and path metrics.

#### Scenario: 2Wiki plan includes learned method
- **WHEN** the 2Wiki tiny workflow is planned with `learned_graph_rgcn_retriever`
- **THEN** the plan includes proposal graph, pairs, train, retrieve, evaluate, and aggregate stages for the learned method

#### Scenario: 2Wiki dependency labels can supervise edge loss
- **WHEN** 2Wiki labels include non-empty `gold_dependency_edges` and matching proposal edges exist
- **THEN** the learned graph training batch includes positive edge labels and a positive edge loss sample count

#### Scenario: 2Wiki path metrics use retrieved subgraph
- **WHEN** learned graph R-GCN predictions are evaluated on 2Wiki path-supported labels
- **THEN** path metrics are computed from `prediction.retrieved_subgraph` and label-side `gold_dependency_edges`

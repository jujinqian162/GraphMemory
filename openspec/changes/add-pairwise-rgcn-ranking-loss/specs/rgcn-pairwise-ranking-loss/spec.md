## ADDED Requirements

### Requirement: Configurable pairwise ranking loss
The system SHALL expose R-GCN loss config fields for task-local pairwise ranking loss, including loss weight, temperature, and negative sample type weights.

#### Scenario: Pairwise loss defaults are disabled
- **WHEN** an R-GCN loss config is created without pairwise-specific overrides
- **THEN** `pairwise_rank_loss_weight` is `0.0`, `pairwise_temperature` is positive, and pairwise negative type weights are available for supported negative sample types

#### Scenario: Learned graph config opts in
- **WHEN** `configs/methods/learned_graph_rgcn_retriever.json` is loaded
- **THEN** its effective loss config enables pairwise ranking loss and preserves edge and sparse loss settings

### Requirement: Task-local weighted pairwise loss
The system SHALL compute pairwise ranking loss only between positive and negative samples that belong to the same task.

#### Scenario: Positive should outrank same-task negative
- **WHEN** a training batch contains one positive and one negative sample for the same task
- **THEN** pairwise loss is `softplus(-(positive_score - negative_score) / temperature)` multiplied by the negative sample type weight

#### Scenario: Cross-task samples are ignored
- **WHEN** a training batch contains positive and negative samples from different tasks only
- **THEN** pairwise loss is zero and the pairwise pair count is zero

#### Scenario: Weighted mean is normalized by pair weights
- **WHEN** a training batch contains multiple valid positive-negative pairs with different negative sample type weights
- **THEN** pairwise loss is the weighted mean of pair losses normalized by the sum of pair weights

### Requirement: Pairwise loss contributes to total R-GCN loss
The system SHALL include pairwise ranking loss in the total R-GCN training loss according to `pairwise_rank_loss_weight`.

#### Scenario: Pairwise weight affects total loss
- **WHEN** pairwise loss weight is non-zero
- **THEN** total loss includes `pairwise_rank_loss_weight * pairwise_rank_loss` in addition to existing rank, edge, and sparse loss terms

#### Scenario: Zero pairwise weight preserves existing objective
- **WHEN** pairwise loss weight is zero
- **THEN** total loss equals the existing rank, edge, and sparse weighted objective

### Requirement: Pairwise training metrics
The system SHALL record pairwise ranking loss diagnostics in R-GCN training metrics and checkpoint metadata.

#### Scenario: Metric record includes pairwise diagnostics
- **WHEN** an R-GCN training epoch completes
- **THEN** the metric record includes pairwise rank loss, pairwise pair count, pairwise loss weight, pairwise temperature, and pairwise negative type weights

#### Scenario: Checkpoint preserves pairwise config
- **WHEN** an R-GCN checkpoint is saved
- **THEN** checkpoint metadata includes the effective pairwise ranking loss config needed to reproduce the training objective

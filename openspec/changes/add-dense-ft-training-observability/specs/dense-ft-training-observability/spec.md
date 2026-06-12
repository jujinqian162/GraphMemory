## ADDED Requirements

### Requirement: Dense-ft emits epoch-level training diagnostics

The dense-ft training system SHALL write `train_metrics.jsonl` rows that allow downstream analysis to distinguish improving, overfitting, and underfitting training runs using only dense-ft-owned artifacts.

#### Scenario: Baseline metric row is recorded before training

- **WHEN** dense-ft training starts with a dev evaluator
- **THEN** the first metric row SHALL use `epoch=0`, `global_step=0`, `train_loss=null`, the configured selected dev metric, `best_epoch=0`, and `best_dev_metric` equal to the epoch-0 selected dev metric

#### Scenario: Epoch metric rows are recorded after each training epoch

- **WHEN** dense-ft training completes an optimizer epoch
- **THEN** the training metrics SHALL include one row for that epoch with one-based `epoch`, cumulative `global_step`, averaged `train_loss`, the configured selected dev metric, `best_epoch`, and `best_dev_metric`

#### Scenario: Metric artifact path remains unchanged

- **WHEN** the unified train script serializes a dense-ft training result
- **THEN** it SHALL continue writing the metric rows to the existing dense-ft `learned/dense_ft/train_metrics.jsonl` artifact path

### Requirement: Dense-ft saves the best dev checkpoint

The dense-ft training system SHALL save the SentenceTransformer model directory selected by the configured dev metric instead of treating the final epoch as implicitly best.

#### Scenario: Baseline remains best when training degrades

- **WHEN** the epoch-0 selected dev metric is better than all epoch-end selected dev metrics
- **THEN** `learned/dense_ft/checkpoints/best_model` SHALL contain the epoch-0 model and the final metric row SHALL report `best_epoch=0`

#### Scenario: Improved epoch replaces best model

- **WHEN** a training epoch produces a selected dev metric that improves over the current best according to `higher_is_better`
- **THEN** `learned/dense_ft/checkpoints/best_model` SHALL be overwritten with that epoch's model and later metric rows SHALL report that epoch as `best_epoch` until a better epoch appears

#### Scenario: Lower-is-better selection is respected

- **WHEN** dense-ft is configured with `higher_is_better=false`
- **THEN** best-epoch comparison SHALL treat lower selected metric values as improvements

### Requirement: Observability stays inside dense-ft training

The dense-ft training observability change SHALL preserve public workflow, delivery, and non-dense-ft training contracts.

#### Scenario: Workflow and delivery do not need new behavior

- **WHEN** dense-ft training emits multiple metric rows
- **THEN** `scripts/train_method.py`, experiment workflow planning, and `scripts/deliver/collect_run_artifacts.py` SHALL continue using their existing artifact contracts without dense-ft-specific branching

#### Scenario: R-GCN training metrics are unchanged

- **WHEN** R-GCN training runs after this change
- **THEN** its metric row schema, checkpoint behavior, and train script integration SHALL remain unchanged

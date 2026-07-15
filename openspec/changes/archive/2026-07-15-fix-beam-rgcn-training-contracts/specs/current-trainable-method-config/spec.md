## MODIFIED Requirements

### Requirement: Canonical trainable method configs
The system SHALL load R-GCN and Dense-FT method configuration from canonical composed YAML files under `configs/method_configs/` using the typed experiment configuration models. Both existing R-GCN configs SHALL include typed decoder, beam-search, beam-training loss, optimizer-phase, and checkpoint-selection settings without introducing another method id.

#### Scenario: Load a current R-GCN config
- **WHEN** configuration resolves `dense_rgcn_graph_retriever` or `dense_ft_rgcn_graph_retriever` with a valid profile
- **THEN** it SHALL return the existing method id with typed encoder, pair, R-GCN model, trainer, decoder, beam, loss, optimizer-phase, and selection values

#### Scenario: Canonical selection preserves current behavior
- **WHEN** either canonical R-GCN method config is loaded
- **THEN** its selection settings SHALL be `best_metric: dev_composite` and `higher_is_better: true`
- **AND** `dev_composite` SHALL mean `0.50 * Full Support@5 + 0.30 * Recall@5 + 0.20 * MRR`

#### Scenario: Legacy trainer learning rate is documented
- **WHEN** either canonical R-GCN YAML is inspected
- **THEN** `trainer.learning_rate` SHALL carry an adjacent comment stating that R-GCN optimizer execution uses `optimizer_phases.decoder_learning_rate` and `optimizer_phases.rgcn_learning_rate`
- **AND** changing only `trainer.learning_rate` SHALL NOT change either optimizer parameter-group learning rate

#### Scenario: Load a current Dense-FT config
- **WHEN** configuration resolves `dense_ft` with a valid profile
- **THEN** it SHALL return a Dense-FT method config without R-GCN beam-decoder fields

#### Scenario: Existing R-GCN method list is unchanged
- **WHEN** the runtime registry and experiment method selection are inspected
- **THEN** they SHALL contain the existing R-GCN method ids and SHALL NOT contain a beam-specific retriever method id

### Requirement: Strict current-only validation
The typed configuration models MUST reject missing required beam or selection fields, unknown beam or selection fields, unsupported selection metric names, invalid beam bounds, mismatched train/inference beam sizes, old aliases, legacy defaults containers, and schema version fields.

#### Scenario: Reject an incomplete beam config
- **WHEN** an R-GCN config omits required decoder, maximum-step, stop, loss, beam-size, or selection settings
- **THEN** configuration loading SHALL fail with a structural validation error

#### Scenario: Reject invalid beam bounds
- **WHEN** beam size is non-positive, maximum steps are outside supported positive bounds, or training and inference beam sizes differ
- **THEN** configuration loading SHALL fail before planning or execution

#### Scenario: Reject an unsupported selection metric
- **WHEN** an R-GCN selection config names a metric outside `dev_composite`, `dev_full_support_at_5`, `dev_full_support_at_10`, `dev_recall_at_5`, `dev_mrr`, and `dev_loss`
- **THEN** configuration loading SHALL fail before planning or execution

#### Scenario: Reject a legacy config
- **WHEN** a config contains a retired alias, defaults container, or schema version field
- **THEN** configuration loading SHALL fail with a structural validation error

## ADDED Requirements

### Requirement: R-GCN checkpoint selection is config-authoritative
The R-GCN stage trainer SHALL pass typed selection settings into model-domain training, and checkpoint selection SHALL use the configured metric name and comparison direction without a hard-coded fallback.

#### Scenario: Stage trainer passes selection settings
- **WHEN** `RgcnGraphRetrieverTrainer` starts either existing R-GCN method
- **THEN** it SHALL pass the resolved `best_metric` and `higher_is_better` values from `settings.selection` into `train_graph_retriever`

#### Scenario: Higher-is-better selection
- **WHEN** `higher_is_better` is true
- **THEN** a checkpoint SHALL replace the current best checkpoint only when the configured selected metric increases

#### Scenario: Lower-is-better selection
- **WHEN** `higher_is_better` is false
- **THEN** a checkpoint SHALL replace the current best checkpoint only when the configured selected metric decreases

#### Scenario: Development loss can drive selection
- **WHEN** `best_metric` is `dev_loss` and `higher_is_better` is false
- **THEN** checkpoint selection SHALL use the complete weighted beam development loss

#### Scenario: Selection provenance is visible
- **WHEN** an R-GCN training metric record is written
- **THEN** it SHALL identify the selected metric name, current selected value, and best selected value used for checkpoint selection

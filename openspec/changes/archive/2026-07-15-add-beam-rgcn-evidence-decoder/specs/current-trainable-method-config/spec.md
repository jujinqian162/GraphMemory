## MODIFIED Requirements

### Requirement: Canonical trainable method configs
The system SHALL load R-GCN and Dense-FT method configuration from canonical composed YAML files under `configs/method_configs/` using the typed experiment configuration models. Both existing R-GCN configs SHALL include typed decoder, beam-search, beam-training loss, and optimizer-phase settings without introducing another method id.

#### Scenario: Load a current R-GCN config
- **WHEN** configuration resolves `dense_rgcn_graph_retriever` or `dense_ft_rgcn_graph_retriever` with a valid profile
- **THEN** it SHALL return the existing method id with typed encoder, pair, R-GCN model, trainer, decoder, beam, loss, and selection values

#### Scenario: Load a current Dense-FT config
- **WHEN** configuration resolves `dense_ft` with a valid profile
- **THEN** it SHALL return a Dense-FT method config without R-GCN beam-decoder fields

#### Scenario: Existing R-GCN method list is unchanged
- **WHEN** the runtime registry and experiment method selection are inspected
- **THEN** they SHALL contain the existing R-GCN method ids and SHALL NOT contain a beam-specific retriever method id

### Requirement: Strict current-only validation
The typed configuration models MUST reject missing required beam fields, unknown beam fields, invalid beam bounds, mismatched train/inference beam sizes, old aliases, legacy defaults containers, and schema version fields.

#### Scenario: Reject an incomplete beam config
- **WHEN** an R-GCN config omits required decoder, maximum-step, stop, loss, or beam-size settings
- **THEN** configuration loading SHALL fail with a structural validation error

#### Scenario: Reject invalid beam bounds
- **WHEN** beam size is non-positive, maximum steps are outside supported positive bounds, or training and inference beam sizes differ
- **THEN** configuration loading SHALL fail before planning or execution

#### Scenario: Reject a legacy config
- **WHEN** a config contains a retired alias, defaults container, or schema version field
- **THEN** configuration loading SHALL fail with a structural validation error

## ADDED Requirements

### Requirement: Canonical trainable method configs
The system SHALL load R-GCN and Dense-FT method configuration from canonical files under `configs/methods/` using a typed trainable method config union.

#### Scenario: Load a current R-GCN config
- **WHEN** the loader receives the canonical R-GCN method config and a valid profile
- **THEN** it returns an R-GCN method config with typed encoder, pair, train, and profile values

#### Scenario: Load a current Dense-FT config
- **WHEN** the loader receives the canonical Dense-FT method config and a valid profile
- **THEN** it returns a Dense-FT method config with typed encoder, pair, train, and profile values

### Requirement: Strict current-only validation
The trainable method config loader MUST reject missing fields, unknown fields, old aliases, legacy defaults containers, and schema version fields.

#### Scenario: Reject a legacy config
- **WHEN** a config contains `schema_version`, `defaults`, or a retired field alias
- **THEN** loading fails with a structural validation error

### Requirement: Experiment method config references
Experiment configuration SHALL reference trainable method configs through `method_configs` and MUST reject the retired `training_configs` key.

#### Scenario: Reject the retired experiment key
- **WHEN** an experiment config contains `training_configs`
- **THEN** experiment config loading fails

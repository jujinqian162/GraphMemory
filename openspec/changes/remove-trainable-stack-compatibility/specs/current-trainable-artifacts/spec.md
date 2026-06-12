## ADDED Requirements

### Requirement: Current R-GCN checkpoint contract
R-GCN training SHALL write and read a strict R-GCN-specific checkpoint contract without a checkpoint version field.

#### Scenario: Load a current checkpoint
- **WHEN** retrieval loads a checkpoint written by the current R-GCN trainer
- **THEN** typed R-GCN model, trainer, encoder, and state records are restored

#### Scenario: Reject a versioned checkpoint
- **WHEN** a checkpoint contains the retired `checkpoint_version` field
- **THEN** checkpoint loading fails

### Requirement: Current Dense-FT metadata contract
Dense-FT training SHALL write and read one typed metadata contract without a schema version field.

#### Scenario: Load current Dense-FT metadata
- **WHEN** retrieval loads a current Dense-FT model directory
- **THEN** model, encoder, device, and training metadata are validated through the shared metadata type

#### Scenario: Reject versioned metadata
- **WHEN** Dense-FT metadata contains the retired `schema_version` field
- **THEN** metadata loading fails

### Requirement: Exhaustive training unions
Training dispatch and result handling MUST explicitly cover every member of the current train stage config union and MUST fail for unsupported members.

#### Scenario: Receive an unsupported train config
- **WHEN** training dispatch receives a config value outside R-GCN and Dense-FT
- **THEN** it raises instead of selecting a default method implementation

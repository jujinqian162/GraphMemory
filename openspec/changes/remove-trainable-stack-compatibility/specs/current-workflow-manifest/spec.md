## ADDED Requirements

### Requirement: Precompiled stage configurations
The workflow compiler SHALL create complete typed stage configuration files before execution, and low-level scripts SHALL accept those files as their sole configuration contract.

#### Scenario: Plan a trainable workflow
- **WHEN** an experiment is initialized for R-GCN or Dense-FT
- **THEN** the run directory contains complete pair, train, retrieve, and evaluate stage config files

#### Scenario: Execute a stage
- **WHEN** the planner creates a low-level stage command
- **THEN** the command consists of the stage script and `--config` pointing to the complete stage config

### Requirement: Strict current manifest
The workflow manifest MUST match one typed current structure with required stage config references and MUST NOT contain a schema version.

#### Scenario: Reject a legacy manifest
- **WHEN** resume reads a manifest with a version field or without required stage configs
- **THEN** manifest validation fails

#### Scenario: Rebuild with force
- **WHEN** the user invokes experiment initialization with `--force`
- **THEN** stale state is replaced by a newly compiled current manifest and stage configs

### Requirement: No legacy command assembly
The planner MUST NOT reconstruct method-specific argv from artifact maps or use fallback behavior when a stage config is absent.

#### Scenario: Missing stage config
- **WHEN** a required stage config reference is absent
- **THEN** planning fails before a command is produced

### Requirement: Typed ablation compilation
Ablation variants SHALL patch current typed method config fields and compile their own stage configs while preserving baseline alias and invalidation semantics.

#### Scenario: Compile an ablation variant
- **WHEN** an R-GCN ablation patch is selected
- **THEN** the patch is applied to current method config fields and variant-specific stage configs are written

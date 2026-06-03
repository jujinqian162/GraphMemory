## ADDED Requirements

### Requirement: Training pair generation is owned by the training-pairs domain
The system SHALL construct train pair artifacts through `graph_memory.training_pairs` rather than `graph_memory.learned.data`.

#### Scenario: Build train pairs through owned domain
- **WHEN** train pair artifacts are built by the CLI or tests
- **THEN** production, script, and test imports SHALL use `graph_memory.training_pairs` for pair generation

### Requirement: Training pair artifacts remain behavior-equivalent
The system SHALL preserve train pair row order, sample types, labels, de-duplication behavior, random sampling behavior, hard negative ordering, graph-neighbor ordering, and summary statistics.

#### Scenario: Pair artifact equivalence
- **WHEN** the fixed tiny pair fixture is built after the refactor
- **THEN** the resulting pair records and summary SHALL match the pre-refactor expected artifact exactly

### Requirement: Negative samplers remain independent of trainable model internals
The system SHALL keep negative samplers independent of `graph_memory.models.graph_retriever` training, inference, and neural model internals.

#### Scenario: Sampler dependency boundary
- **WHEN** the training-pairs modules are inspected
- **THEN** they SHALL NOT import trainable graph model training, inference, checkpoint, or neural modules

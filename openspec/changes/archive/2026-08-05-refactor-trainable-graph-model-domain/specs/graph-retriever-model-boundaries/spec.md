## ADDED Requirements

### Requirement: Trainable graph retriever internals are owned by the model domain
The system SHALL place trainable graph retriever config, checkpoint, tensorization, batching, neural components, model factory, training, dev evaluation, text embeddings, and inference under `graph_memory.models.graph_retriever`.

#### Scenario: Model-domain imports
- **WHEN** trainable graph retriever scripts and tests import model internals
- **THEN** they SHALL import owned modules from `graph_memory.models.graph_retriever`

### Requirement: Inference does not depend on training
The system SHALL make model inference depend on a shared model factory rather than importing training lifecycle code.

#### Scenario: Inference import boundary
- **WHEN** `graph_memory.models.graph_retriever.inference` is inspected
- **THEN** it SHALL NOT import `graph_memory.models.graph_retriever.training`

### Requirement: Trainable retrieval adapter is retrieval-owned
The system SHALL adapt checkpoint-backed trainable graph retrieval into the public retrieval contract through `graph_memory.retrieval.methods.trainable_graph`.

#### Scenario: Factory builds trainable graph method
- **WHEN** the retrieval factory resolves `dense_rgcn_graph_retriever`
- **THEN** it SHALL construct the retrieval-owned adapter while preserving ranking output and retrieved subgraph semantics

### Requirement: Trainable graph model behavior remains equivalent
The system SHALL preserve relation vocab ordering, tensorization order and dtypes, model forward math, checkpoint schema, ablation mapping, one-step CPU training behavior, and checkpoint-backed retrieval ranking behavior.

#### Scenario: Model behavior equivalence
- **WHEN** focused trainable graph retriever tensorization, model, training, checkpoint, and retrieval tests run after the refactor
- **THEN** their expected tensors, logits, state transitions, checkpoint metadata, and ranking invariants SHALL remain unchanged

### Requirement: Workflow integration port remains narrow
The system SHALL keep `graph_memory.training_config` available for workflow-facing trainable configuration loading without turning old learned modules into broad compatibility facades.

#### Scenario: Workflow-facing config compatibility
- **WHEN** workflow and script config loading paths are exercised
- **THEN** public trainable config fields, profile resolution, device resolution, and parser contracts SHALL remain unchanged

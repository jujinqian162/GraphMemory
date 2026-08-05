## ADDED Requirements

### Requirement: Config loader owns profile application
The system SHALL apply config profiles only inside the config loading boundary before returning a structured stage config to scripts or domain code.

#### Scenario: Stage without profile selector
- **WHEN** a stage config spec does not define a profile selector
- **THEN** config loading MUST skip profile lookup and return the effective config from raw, registry, and CLI patches only

#### Scenario: Stage with profile selector
- **WHEN** a stage config spec defines a profile selector and the raw config contains profiles
- **THEN** config loading MUST merge the selected profile before registry and CLI patches

### Requirement: Runtime config conversions are centralized
The system SHALL expose named adapter functions for converting registry settings into runtime configs instead of duplicating private conversion helpers in scripts or dataclass methods.

#### Scenario: Dense encoder settings conversion
- **WHEN** pair building requires a dense runtime config
- **THEN** the script MUST use the shared conversion adapter from `DenseEncoderSettings` to `DenseConfig`

#### Scenario: Trainer settings conversion
- **WHEN** the training registry builds an R-GCN trainer
- **THEN** it MUST use the shared conversion adapter from trainer settings to `TrainableTrainingConfig`

### Requirement: Retrieval builders validate concrete payloads
The retrieval registry SHALL accept an object payload at its public build boundary and each concrete builder MUST validate that payload against the payload type it requires before constructing a method.

#### Scenario: Correct payload
- **WHEN** a graph-rerank method is built with a `GraphRerankBuildPayload`
- **THEN** the builder MUST construct the method using the typed payload fields

#### Scenario: Incorrect payload
- **WHEN** a graph-rerank method is built with an incompatible payload object
- **THEN** the builder MUST raise a clear error naming the expected payload type

### Requirement: Checkpoint loader receives assembled dependencies
Checkpoint graph retriever loading SHALL require already assembled text and seed providers instead of creating default retrieval dependencies inside the loader.

#### Scenario: Checkpoint method build
- **WHEN** the registry builds a checkpoint graph retrieval method without injected providers
- **THEN** the registry builder MUST assemble default text and seed providers before calling the checkpoint loader

### Requirement: Train labels are optional but used when provided
The train graph retriever CLI SHALL allow train labels to be omitted and SHALL use them for train-pair validation when provided.

#### Scenario: Train labels omitted
- **WHEN** `scripts/train_graph_retriever.py` is called without `--train_labels`
- **THEN** training MUST continue without train-label pair validation

#### Scenario: Train labels provided
- **WHEN** `scripts/train_graph_retriever.py` is called with `--train_labels`
- **THEN** training MUST read the labels and validate train pairs against them

### Requirement: Retrieval methods return structured results
Retrieval methods SHALL return a structured result containing ranked nodes and retrieval trace data instead of a tuple.

#### Scenario: Flat retrieval result
- **WHEN** a flat retrieval method ranks a task
- **THEN** the result MUST contain ranked nodes and an empty trace

#### Scenario: Graph retrieval result
- **WHEN** a graph-aware retrieval method ranks a task
- **THEN** the result MUST contain ranked nodes and retrieved graph edges in the trace

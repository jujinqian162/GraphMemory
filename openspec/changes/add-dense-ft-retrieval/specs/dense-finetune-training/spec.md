## ADDED Requirements

### Requirement: Dense fine-tuning builds training rows from task pair artifacts
The system SHALL build dense-ft training examples from task inputs, labels, and train-pair records without reading graph tensors as model input.

#### Scenario: Positive-only rows are produced when no negatives exist
- **WHEN** a task has positive train-pair records and no negative train-pair records
- **THEN** the dense-ft data builder emits rows containing `anchor` and `positive` text for each positive record

#### Scenario: Hard negatives are selected deterministically
- **WHEN** a task has positive and negative train-pair records
- **THEN** the dense-ft data builder emits at most the configured number of negatives per positive using the priority order `hard_dense`, `hard_bm25`, `hard_graph_neighbor`, then `easy_random`, preserving original order within the same priority

#### Scenario: Unknown node ids fail before training
- **WHEN** a positive or negative train-pair node id is not present in the corresponding task memory items
- **THEN** the dense-ft data builder raises a validation error naming the task id and missing node id

### Requirement: Dense fine-tuning uses shared dense text formatting
The system SHALL use the same dense query and passage formatting helpers for frozen dense inference and dense-ft training data construction.

#### Scenario: Training and inference text contracts match
- **WHEN** dense-ft examples are built for the same task and memory item that frozen dense inference encodes
- **THEN** the query and passage texts exactly match the texts produced by the shared dense encoding service

### Requirement: Dense fine-tuning builds IR evaluator payloads with task-qualified corpus ids
The system SHALL build dev IR evaluator payloads using task ids for queries and `<task_id>::<node_id>` ids for corpus documents.

#### Scenario: Node ids are unique across tasks
- **WHEN** two tasks contain memory items with the same node id
- **THEN** the IR evaluator corpus ids remain distinct by prefixing each node id with its task id

#### Scenario: Relevant docs use gold evidence nodes
- **WHEN** dev labels contain gold evidence node ids for a task
- **THEN** the IR evaluator relevant-doc mapping points that task query to the corresponding task-qualified corpus ids

### Requirement: Dense fine-tuning writes reusable model metadata
The system SHALL write `dense_ft_model_config.json` beside the saved SentenceTransformer model directory.

#### Scenario: Metadata records inference-critical settings
- **WHEN** dense-ft training completes
- **THEN** the metadata contains schema version, method id, base model, query prefix, passage prefix, encoder batch size, and selected metric configuration

#### Scenario: Saved model directory is loadable by retrieval
- **WHEN** dense-ft training succeeds
- **THEN** the output model directory contains the SentenceTransformer artifacts and dense-ft metadata required by the retrieval builder

### Requirement: Dense fine-tuning integrates SentenceTransformers trainer settings
The system SHALL construct SentenceTransformers training arguments from dense-ft method settings.

#### Scenario: CPU smoke training is explicit
- **WHEN** dense-ft trainer settings use `device` equal to `cpu`
- **THEN** the SentenceTransformers training arguments use CPU execution rather than assuming CUDA

#### Scenario: Trainer hyperparameters come from config
- **WHEN** dense-ft training starts
- **THEN** batch sizes, epochs, learning rate, warmup ratio, max grad norm, logging steps, checkpoint retention, fp16, and bf16 are read from the resolved method config

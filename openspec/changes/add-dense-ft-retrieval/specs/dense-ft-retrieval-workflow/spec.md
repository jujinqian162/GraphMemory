## ADDED Requirements

### Requirement: Dense-ft is registered as a retrieval method
The system SHALL expose `dense_ft` as a public retrieval method with metadata that reflects a learned dense model directory.

#### Scenario: Method list includes dense-ft
- **WHEN** experiment methods are listed
- **THEN** `dense_ft` appears as a selectable method

#### Scenario: Dense-ft metadata requires a checkpoint directory
- **WHEN** dense-ft method metadata is inspected
- **THEN** it declares checkpoint/model-directory requirement, dense encoder usage, no graph-config requirement, and no graph-input requirement for retrieval

### Requirement: Dense-ft retrieval loads metadata from checkpoint directory
The system SHALL build dense-ft retrieval by reading `dense_ft_model_config.json` from the supplied checkpoint directory.

#### Scenario: Metadata configures dense retriever
- **WHEN** dense-ft retrieval receives a valid checkpoint directory
- **THEN** the retrieval builder constructs `DenseTaskRetriever` with `DenseConfig.model_name` pointing to that directory and prefixes/batch size from metadata

#### Scenario: Missing metadata fails clearly
- **WHEN** dense-ft retrieval receives a checkpoint directory without `dense_ft_model_config.json`
- **THEN** the error message names the missing metadata file and the checkpoint path

### Requirement: Dense-ft workflow includes train and retrieve stages
The system SHALL provide a dense-ft experiment workflow that runs prepare, graph building, pair building, training, retrieval, evaluation, and aggregation.

#### Scenario: Dense-ft plan creates full stage commands
- **WHEN** an experiment plan is created for `dense_ft`
- **THEN** the plan contains pairs, train, retrieve, evaluate, and aggregate commands using dense-ft artifacts

#### Scenario: Dense-ft retrieve uses model directory checkpoint
- **WHEN** dense-ft retrieve commands are generated
- **THEN** they pass the learned dense-ft model directory through the checkpoint argument and do not require the user to repeat the encoder model path

### Requirement: Dense-ft artifacts and docs identify checkpoint as model directory
The system SHALL document and manifest dense-ft learned artifacts as SentenceTransformer model directories even when the workflow role is named checkpoint.

#### Scenario: Manifest points to dense-ft model directory
- **WHEN** dense-ft workflow artifacts are built
- **THEN** the learned checkpoint path points to `learned/dense_ft/checkpoints/best_model`

#### Scenario: Operations docs explain dense-ft checkpoint semantics
- **WHEN** users read dense-ft commands documentation
- **THEN** it states that dense-ft checkpoint refers to a SentenceTransformer model directory rather than a `.pt` file

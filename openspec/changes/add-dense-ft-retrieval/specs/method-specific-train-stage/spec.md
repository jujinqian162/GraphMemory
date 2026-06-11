## ADDED Requirements

### Requirement: Train stage config is method-specific at the root
The system SHALL load TRAIN stage configs as a root-level discriminated union keyed by the retrieval method id.

#### Scenario: R-GCN train config preserves graph IO
- **WHEN** a TRAIN stage config declares `method` as `dense_rgcn_graph_retriever`
- **THEN** config loading returns an R-GCN train stage config that includes train/dev graph inputs and graph checkpoint outputs

#### Scenario: Dense-ft train config omits graph IO
- **WHEN** a TRAIN stage config declares `method` as `dense_ft`
- **THEN** config loading returns a dense-ft train stage config that includes task, label, pair, model directory, metrics, and run-summary IO without requiring train/dev graph paths

### Requirement: Training registry dispatches by method settings
The system SHALL dispatch training jobs through the training registry without requiring global R-GCN provider dependencies for every method.

#### Scenario: Dense-ft dispatch does not construct R-GCN providers
- **WHEN** a dense-ft train stage is executed
- **THEN** scripts and the generic train stage runner do not construct `DenseGraphFeatureProvider` or `RetrieverSeedSignalProvider`

#### Scenario: R-GCN dispatch remains available
- **WHEN** an R-GCN train stage is executed
- **THEN** the registry builds the R-GCN trainer and constructs the graph feature and seed-signal dependencies inside the R-GCN-specific path

### Requirement: One canonical train script runs all training methods
The system SHALL use `scripts/train_method.py --method <method>` as the train CLI entry for all workflow-generated and direct low-level training commands.

#### Scenario: R-GCN train command uses unified entry
- **WHEN** the workflow builds an R-GCN train command
- **THEN** the command invokes `scripts/train_method.py --method dense_rgcn_graph_retriever`

#### Scenario: Dense-ft train command uses unified entry
- **WHEN** the workflow builds a dense-ft train command
- **THEN** the command invokes `scripts/train_method.py --method dense_ft`

#### Scenario: Old graph train script is removed
- **WHEN** repository scripts and tests are searched for active train commands
- **THEN** they no longer reference `scripts/train_graph_retriever.py` except in historical documentation

### Requirement: Training profiles are override-only
The system SHALL treat training config profiles as overrides of `defaults` and SHALL avoid repeating default values in profile bodies.

#### Scenario: CUDA default is not repeated in CUDA profiles
- **WHEN** dense-ft default trainer settings specify `device` as `cuda`
- **THEN** profiles such as `quick` and `cloud-full` do not repeat `device: "cuda"` unless they intentionally differ from the default

#### Scenario: CPU smoke profile overrides device
- **WHEN** the dense-ft smoke profile is resolved
- **THEN** it explicitly overrides trainer device to `cpu` because that differs from the CUDA default

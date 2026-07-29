## ADDED Requirements

### Requirement: Training inputs are validated before expensive work
Dense-FT, evidence R-GCN, and provenance R-GCN training entrypoints SHALL receive typed Pydantic configuration and scientific input aggregates. All schema, task alignment, candidate/label/graph/pair membership, supervision, and configuration cross-field validation required to begin the run MUST finish before loading a trainable model, encoding the full corpus, materializing training tensors, creating an optimizer, or entering the first epoch.

#### Scenario: Training pair artifact is invalid
- **WHEN** a training pair references an unknown or gold-negative node
- **THEN** training fails in preflight before encoder/model initialization and epoch zero

#### Scenario: Dev labels and requests are misaligned
- **WHEN** the dev split has missing or extra labels relative to requests
- **THEN** training fails in preflight before expensive feature materialization

#### Scenario: Validated inputs begin training
- **WHEN** all input aggregates and configuration models pass
- **THEN** the trainer does not repeat detached schema validators after training has started

### Requirement: Domain training configuration has one source of truth
R-GCN model configuration, training configuration, feature configuration, negative-sampling configuration, provenance loss/inference/selection configuration, and Dense-FT metadata SHALL use authoritative low-level Pydantic models. Experiment configuration and stage adapters SHALL compose or reuse those models rather than mirror their fields and defaults in validation-bearing dataclasses.

#### Scenario: Invalid dropout is configured
- **WHEN** dropout is outside the active model's accepted range
- **THEN** Pydantic rejects the configuration before model construction

#### Scenario: Invalid optimizer or graph policy is configured
- **WHEN** an unsupported optimizer, graph encoder, message transform, edge-weight policy, relation, feature name, or ablation value is supplied
- **THEN** the owning configuration model rejects it before training

#### Scenario: Batch field changes
- **WHEN** a graph-batch configuration field changes in the authoritative model
- **THEN** experiment composition, trainer construction, tracking, and checkpoint serialization consume that same model field

### Requirement: Dev predictions are validated before checkpoint selection
Every epoch's dev predictions SHALL use the production Pydantic ranked-result boundary and SHALL be valid before metric calculation, selection metric calculation, best-state replacement, tracking, or checkpoint callbacks.

#### Scenario: Dev result violates ranking coverage
- **WHEN** dev inference fails to rank every request candidate exactly once
- **THEN** the epoch fails before its metric row can influence best-checkpoint selection

#### Scenario: Dev metric row is malformed
- **WHEN** a dev evaluator emits a non-finite, out-of-range, wrong-schema, or incomplete metric row
- **THEN** the metric suite model rejects it before the selection policy consumes it

### Requirement: Checkpoint envelopes are Pydantic contracts
Evidence R-GCN and provenance R-GCN checkpoint families SHALL use closed Pydantic envelope/metadata models for schema version, method and variant identity, epoch/global-step counters, finite selection metrics, model/training/configuration models, creation identity, and required state-map presence. Dense-FT metadata SHALL likewise use a Pydantic model rather than TypeAdapter validation of an unvalidated dataclass.

#### Scenario: Checkpoint is saved
- **WHEN** a trainer prepares a checkpoint
- **THEN** a Pydantic checkpoint envelope validates all serializable metadata and required state-map slots before `torch.save`

#### Scenario: Checkpoint is loaded
- **WHEN** inference or resume loads a checkpoint
- **THEN** the same checkpoint envelope model validates it before model reconstruction

#### Scenario: Method identity mismatches
- **WHEN** a checkpoint method/variant/schema does not match the requested model family
- **THEN** the checkpoint model rejects it without a separate `_validate_payload` function or copied field set

### Requirement: Torch state internals remain explicitly opaque
Pydantic checkpoint envelopes SHALL validate state-map presence and outer type but SHALL NOT recursively coerce or serialize PyTorch tensors, optimizer internals, or scheduler internals. The owning trainer/framework remains responsible for state-dict key/shape compatibility during model or optimizer loading.

#### Scenario: Tensor state is present
- **WHEN** a checkpoint contains tensor state maps with valid metadata
- **THEN** Pydantic preserves the state objects without attempting JSON conversion or numerical coercion

#### Scenario: State dict is incompatible with the model
- **WHEN** framework loading finds missing or unexpected tensor keys
- **THEN** the model loader reports the framework compatibility error after the envelope itself has passed

### Requirement: No schema validation is deferred solely to checkpoint save or final retrieval
A condition knowable from configuration or training inputs MUST NOT first be checked after the epoch loop. A condition knowable from one result MUST NOT first be checked after every retrieval task has run. Only aggregate completeness conditions that inherently require the full collection MAY remain at collection completion.

#### Scenario: Invalid configuration reaches checkpoint save
- **WHEN** a model or training configuration violates an active field/cross-field invariant
- **THEN** tests demonstrate that training preflight fails and checkpoint save is never the first detector

#### Scenario: Invalid first retrieval result is produced
- **WHEN** the first task returns an invalid ranking or trace
- **THEN** later retrieval tasks are not executed

### Requirement: Training output records are typed before publication
Training history rows, selected metric metadata, pair summaries, and model result metadata that are persisted or included in artifact identities SHALL have Pydantic/JSON-value contracts and finite numeric checks before publication.

#### Scenario: Training history contains NaN
- **WHEN** a loss, gradient norm, throughput, memory count, dev metric, or selection value intended for persistence is non-finite where the contract requires a finite value
- **THEN** the output model rejects publication and reports the offending field location

## ADDED Requirements

### Requirement: Memory Stream is a selectable public retrieval method
The system SHALL later register `memory_stream` with a dense encoder dependency,
a read-only global importance dependency, and no retrieval-time graph,
tuned-config, or checkpoint dependency.

#### Scenario: Method listing includes Memory Stream
- **WHEN** the retrieval milestone is complete and experiment methods are listed
- **THEN** `memory_stream` appears as a selectable method

#### Scenario: Method definition describes exact dependencies
- **WHEN** Memory Stream method metadata is inspected
- **THEN** it declares experiment-config dense encoding and an external global importance artifact

### Requirement: Importance preparation is outside workflow execution
The system SHALL NOT register importance annotation as a workflow stage or
allocate its output inside a workflow run directory.

#### Scenario: Full plan contains no annotation command
- **WHEN** an experiment plan is built for `memory_stream`
- **THEN** its stages are prepare, graph construction, retrieval, evaluation, and aggregation
- **AND** no `scripts/annotate_importance.py` command is present

#### Scenario: Manifest does not own global importance
- **WHEN** a Memory Stream experiment is initialized
- **THEN** the manifest does not allocate a run-local importance artifact, annotation summary, or annotation stage config

### Requirement: Retrieval consumes a global external importance artifact
The system SHALL require a complete global importance artifact before Memory
Stream retrieval and SHALL treat it as a read-only external dependency.

#### Scenario: Missing global artifact blocks retrieval
- **WHEN** Memory Stream retrieval is planned and the configured global importance artifact is missing
- **THEN** planning or retrieval fails with the missing path

#### Scenario: Workflow subset is selected from the global artifact
- **WHEN** workflow tasks are a subset of the canonical annotated corpus
- **THEN** records are joined by task id and validated by content digest and exact node coverage

#### Scenario: Extra global tasks are accepted
- **WHEN** the global artifact contains canonical tasks not selected by the workflow profile
- **THEN** retrieval accepts the artifact and ignores unselected tasks

### Requirement: Workflow configuration excludes annotation runtime settings
The system SHALL keep annotation model, cache, generation, and IO settings out of
experiment workflow configuration.

#### Scenario: Retrieval scoring settings are valid
- **WHEN** Memory Stream weights, recency decay, dense encoder, and external importance path are valid
- **THEN** workflow initialization may write a retrieve stage config

#### Scenario: Annotation settings are not compiled
- **WHEN** workflow initialization selects Memory Stream
- **THEN** it does not validate or compile model path, prompt, cache directory, device, or generation settings

### Requirement: Status and delivery record but do not own global importance
The system SHALL report the external dependency used by retrieval without
treating annotation as a resumable workflow stage.

#### Scenario: Status has no importance stage row
- **WHEN** Memory Stream experiment status is displayed
- **THEN** no importance stage status is reported

#### Scenario: Delivery records reproducibility evidence
- **WHEN** a Memory Stream run is collected
- **THEN** retrieval provenance identifies the global importance path and semantic metadata
- **AND** the global cache is not copied into the run delivery

### Requirement: Operations documentation covers one-time MetaX preprocessing
The system SHALL document the zero-argument global command, model acquisition,
direct Transformers loading, environment preparation, restart behavior, and
later retrieval consumption.

#### Scenario: Operator runs the default command
- **WHEN** canonical dev input and the default local model path exist
- **THEN** `python scripts/annotate_importance.py` starts global annotation without a config file

#### Scenario: Documentation preserves one long-lived model instance
- **WHEN** annotation contains many cache misses
- **THEN** one process selects one GPU, loads the model once, processes all misses, and exits after the global artifact is complete

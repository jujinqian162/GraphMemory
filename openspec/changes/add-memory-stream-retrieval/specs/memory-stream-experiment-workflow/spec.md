## ADDED Requirements

### Requirement: Memory Stream is a selectable public retrieval method
The system SHALL register `memory_stream` with a distinct lifecycle, dense encoder dependency, sidecar importance dependency, and no retrieval-time graph, tuned-config, or checkpoint dependency.

#### Scenario: Method listing includes Memory Stream
- **WHEN** experiment methods are listed
- **THEN** `memory_stream` appears as a selectable method

#### Scenario: Method definition describes exact dependencies
- **WHEN** Memory Stream method metadata is inspected
- **THEN** it declares experiment-config dense encoding and an importance sidecar while declaring no graph input, graph config, model checkpoint, or train artifact

### Requirement: Memory Stream has a dedicated experiment workflow
The system SHALL plan Memory Stream as prepare, graph construction, importance annotation, retrieval, evaluation, and aggregation.

#### Scenario: Full plan includes importance before retrieval
- **WHEN** an experiment plan is built for `memory_stream`
- **THEN** the importance command appears after graph construction and before retrieval

#### Scenario: Importance command uses a stage-root config
- **WHEN** workflow commands are rendered
- **THEN** annotation invokes `scripts/annotate_importance.py --config <path>` without reconstructing method-specific CLI arguments

#### Scenario: Train and pair stages are absent
- **WHEN** required stages are computed for only `memory_stream`
- **THEN** pair building and training are not included

### Requirement: Workflow artifacts separate annotation from predictions
The system SHALL allocate a method-specific importance artifact and run summary independently from Memory Stream prediction and metric artifacts.

#### Scenario: Manifest exposes importance paths
- **WHEN** a Memory Stream experiment is initialized
- **THEN** the manifest contains an importance sidecar path, annotation run-summary path, prediction path, metric path, and corresponding stage-config paths

#### Scenario: Retrieve stage consumes the compiled importance path
- **WHEN** the Memory Stream retrieve stage config is written
- **THEN** its IO references the manifest importance artifact produced by the annotation stage

### Requirement: Planner dependencies require a complete importance artifact
The system SHALL prevent a Memory Stream retrieve-only plan when the importance stage is omitted and no complete non-stale importance artifact exists.

#### Scenario: Missing importance blocks retrieval
- **WHEN** retrieval is selected without the importance stage and the sidecar is missing
- **THEN** planning fails with the missing importance path

#### Scenario: Completed importance permits retrieval-only execution
- **WHEN** the sidecar and matching successful run summary exist
- **THEN** retrieval may be planned without rerunning annotation

#### Scenario: Changed annotation settings mark importance stale
- **WHEN** model id, prompt version, generation settings, task input path, or cache semantics differ from the successful run summary
- **THEN** status marks the importance stage stale and cache-aware planning does not treat it as complete

#### Scenario: Changed memory content marks importance stale
- **WHEN** the task path is unchanged but memory item id, source, text, position, or order differs from the annotated artifact
- **THEN** status marks the importance stage stale and requires annotation before retrieval

### Requirement: Experiment configuration validates Memory Stream settings
The system SHALL require a complete fixed Memory Stream configuration when the method is selected.

#### Scenario: Valid configuration compiles both stages
- **WHEN** model id, model path, prompt version, cache directory, device/loading settings, generation settings, dense encoder, weights, and recency decay are valid
- **THEN** workflow initialization writes typed importance and retrieve stage configs

#### Scenario: Invalid scoring settings fail during initialization
- **WHEN** a weight is negative, all weights are zero, or recency decay is outside `(0, 1]`
- **THEN** experiment initialization fails before writing an executable manifest

#### Scenario: Invalid annotation settings fail during initialization
- **WHEN** model id, model path, or prompt version is empty, device is unsupported, max new tokens is non-positive, or deterministic generation settings are violated
- **THEN** experiment initialization fails before writing an executable manifest

### Requirement: Status, resume, and delivery include the importance stage
The system SHALL inspect, cache-prune, resume, and collect Memory Stream annotation artifacts using the same current-only workflow semantics as other stages.

#### Scenario: Successful annotation is complete
- **WHEN** the sidecar exists and its run summary matches expected tasks, outputs, model, prompt, and generation settings
- **THEN** experiment status reports `importance memory_stream complete`

#### Scenario: Missing summary is stale
- **WHEN** the sidecar exists without a matching successful run summary
- **THEN** experiment status reports the importance stage as stale

#### Scenario: Delivery collector includes reproducibility evidence
- **WHEN** a Memory Stream run is collected for reporting
- **THEN** the importance sidecar, annotation run summary, prediction summary, metrics, effective config, and stage configs are included

### Requirement: Operations documentation covers MetaX offline execution
The system SHALL document model acquisition, direct Transformers loading, process environment preparation, persistent model lifecycle, annotation execution, restart behavior, and retrieval execution for the MetaX C500 environment.

#### Scenario: Documentation uses the proven direct invocation path
- **WHEN** an operator follows the Memory Stream instructions
- **THEN** the annotation process imports `AutoTokenizer` and `AutoModelForCausalLM` in the proven vendor-compatible environment without starting an HTTP server

#### Scenario: Documentation preserves one long-lived model instance
- **WHEN** annotation contains many cache misses
- **THEN** the instructions explain that one process selects one GPU, loads the model once, processes all misses, and exits only after the stage completes

#### Scenario: Documentation explains restart cost
- **WHEN** the annotation process fails and is restarted
- **THEN** the instructions explain that the model is loaded once again but successful per-task cache entries prevent completed generations from repeating

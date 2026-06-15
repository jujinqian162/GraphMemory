## ADDED Requirements

### Requirement: Memory Stream reuses the stateless retrieval workflow
The system SHALL register Memory Stream with the stateless retrieval lifecycle
and SHALL reuse the existing stateless retrieval workflow with a dense encoder
and read-only global importance dependency.

#### Scenario: Existing workflow is selected
- **WHEN** a Memory Stream experiment is initialized
- **THEN** the method registry selects `STATELESS_RETRIEVAL_WORKFLOW`
- **AND** no new workflow id or stage id is required
- **AND** it contains no importance annotation command or stage

#### Scenario: Manifest does not own global importance
- **WHEN** the workflow manifest is compiled
- **THEN** it does not allocate a run-local importance artifact or cleaning summary

### Requirement: Retrieval consumes a global compact importance artifact
The system SHALL require a valid schema-versioned importance artifact before
Memory Stream retrieval starts.

#### Scenario: Missing global artifact fails
- **WHEN** Memory Stream retrieval is planned and the configured importance path is missing
- **THEN** planning or retrieval fails with that concrete path

#### Scenario: Default path targets cleaned first-1000 data
- **WHEN** no importance path override is supplied
- **THEN** the workflow uses `data/hotpotqa/processed/memory_stream/dev.first_1000.importance.json`

#### Scenario: Experiment path override is explicit
- **WHEN** `memory_stream_importance_path` is configured
- **THEN** the Memory Stream retrieve-stage config uses that path instead of the default

#### Scenario: Retrieve stage owns the external path
- **WHEN** a Memory Stream retrieve-stage config is compiled
- **THEN** `RetrieveIO.importance` contains the external artifact path
- **AND** the path is not represented as a run-local manifest artifact

### Requirement: Memory Stream may cap the test split to available coverage
The system SHALL allow Memory Stream to cap its test split to the cleaned
importance coverage when the selected profile requests more examples than the
artifact can support, and SHALL emit a warning instead of failing.

#### Scenario: Cloud-full profile is truncated for Memory Stream only
- **WHEN** the selected profile asks for more test examples than the cleaned Memory Stream prefix covers
- **THEN** workflow planning caps the Memory Stream test split to the available covered prefix
- **AND** the workflow emits a warning naming the capped count
- **AND** other methods keep their normal shared split policy
- **AND** the capped count is used by both retrieve and evaluate stage configs

### Requirement: Experiment config excludes production settings
The system SHALL configure only retrieval weights, recency decay, dense
encoder settings, and an optional importance path override.

#### Scenario: Retrieve config compilation
- **WHEN** Memory Stream retrieval settings are compiled
- **THEN** the job contains encoder, weights, recency decay, and top-k
- **AND** the IO contains the importance path
- **AND** no annotation model path, prompt, cache directory, device, or generation setting is accepted

### Requirement: Status and delivery record but do not own importance
The system SHALL keep workflow status free of an importance stage and SHALL
record the external artifact as retrieval provenance.

#### Scenario: Status has no importance stage row
- **WHEN** workflow status is rendered
- **THEN** no importance stage status is reported

#### Scenario: Delivery excludes external data
- **WHEN** a Memory Stream run is delivered
- **THEN** retrieval provenance identifies the external importance path and hash
- **AND** the global artifact and cleaning summary are not copied into run delivery

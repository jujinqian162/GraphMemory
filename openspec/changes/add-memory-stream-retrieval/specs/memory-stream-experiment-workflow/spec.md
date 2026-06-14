## ADDED Requirements

### Requirement: Memory Stream is a retrieval-only workflow
The system SHALL expose Memory Stream as a retrieval workflow with dense
encoding and a read-only global importance dependency, without an importance
production stage.

#### Scenario: Workflow plan contains retrieval only
- **WHEN** a Memory Stream experiment is initialized
- **THEN** its plan contains the required retrieval work
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

### Requirement: Experiment config excludes production settings
The system SHALL configure only retrieval weights, recency decay, dense
encoder settings, and an optional importance path override.

#### Scenario: Retrieve config compilation
- **WHEN** Memory Stream retrieval settings are compiled
- **THEN** no model path, prompt, cache directory, device, or generation setting is accepted

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

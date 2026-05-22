## ADDED Requirements

### Requirement: Named experiment runs

The system SHALL provide a user-facing experiment runner that creates and uses a named run directory under `runs/<experiment_name>/` for all run-specific artifacts.

#### Scenario: Create isolated run directory

- **WHEN** a user initializes or runs experiment `quick_valid_100`
- **THEN** the system creates or uses `runs/quick_valid_100/` as the only default location for run-specific inputs, graphs, tuned configs, predictions, metrics, tables, debug artifacts, summaries, and manifest files

#### Scenario: Do not write run-specific selected configs globally

- **WHEN** graph-rerank tuning runs inside an experiment
- **THEN** the selected graph-rerank config is written under `runs/<experiment_name>/tuned/`
- **THEN** global `configs/` files are not modified by default

### Requirement: Manifest-based run provenance

The system SHALL maintain a `manifest.json` file in each run directory that records the effective experiment intent and generated artifact paths.

#### Scenario: Manifest records effective run configuration

- **WHEN** a run is initialized or executed
- **THEN** `manifest.json` records the experiment name, recipe, profile, selected stages, selected methods, effective config, and generated artifact paths

#### Scenario: Manifest supports repeated commands

- **WHEN** a user runs a later command against an existing experiment name
- **THEN** the system reads the existing manifest and reuses its generated artifact paths unless the user explicitly requests a reset or config-changing reinitialization

### Requirement: Clear config roles

The system SHALL separate stable experiment defaults, tuning search spaces, published selected configs, and run-local outputs into distinct config locations and artifact locations.

#### Scenario: Load experiment defaults

- **WHEN** a user runs an experiment without specifying a config path
- **THEN** the system loads the configured experiment defaults from `configs/experiments/`

#### Scenario: Load tuning search space

- **WHEN** a tuning stage needs graph-rerank candidate values
- **THEN** the system reads those candidate values from `configs/search_spaces/`

#### Scenario: Preserve published configs

- **WHEN** an ordinary experiment run produces tuned parameters
- **THEN** the system does not overwrite configs under `configs/published/`

### Requirement: Config precedence and effective config recording

The system SHALL merge configuration with the precedence `CLI overrides > experiment config/profile > code defaults` and SHALL persist the merged effective config in the run directory.

#### Scenario: CLI override wins

- **WHEN** a user runs an experiment with a profile and a CLI override for a supported setting
- **THEN** the effective config uses the CLI override value for that setting

#### Scenario: Effective config is persisted

- **WHEN** config merging completes for a run
- **THEN** the system writes the merged config to `runs/<experiment_name>/config/effective_config.json`
- **THEN** the manifest references or embeds that effective config

### Requirement: Stage planning and execution

The system SHALL support stage-based planning and execution for the current evidence-retrieval workflow without requiring users to manually provide low-level input and output paths.

#### Scenario: Plan full workflow

- **WHEN** a user asks the runner to plan all stages for an experiment
- **THEN** the system lists the low-level commands for `prepare`, `graphs`, `tune`, `retrieve`, `evaluate`, and `aggregate` using paths generated from the manifest
- **THEN** the system does not execute those commands

#### Scenario: Execute selected stages

- **WHEN** a user asks the runner to execute selected stages
- **THEN** the system executes only those stages and their required dependencies according to the recipe

#### Scenario: Resume from a stage

- **WHEN** a user asks the runner to run from `retrieve`
- **THEN** the system skips earlier completed stages whose artifacts match the manifest
- **THEN** the system runs `retrieve` and later requested stages using manifest paths

### Requirement: Method selection

The system SHALL allow users to select one or more retrieval methods for stages that are method-specific.

#### Scenario: Run selected methods

- **WHEN** a user requests methods `dense,dense_graph_rerank`
- **THEN** method-specific retrieval and evaluation stages run for `dense` and `dense_graph_rerank`
- **THEN** method-specific outputs are written with deterministic method-based filenames under the run directory

#### Scenario: Reject unknown method

- **WHEN** a user requests a method name not supported by the current experiment recipe
- **THEN** the runner fails fast with a clear error before running low-level commands

### Requirement: Preserve low-level contracts

The system SHALL call existing low-level scripts with explicit input and output arguments instead of replacing their artifact contract boundaries.

#### Scenario: Generated command includes explicit paths

- **WHEN** the runner plans or executes a graph construction stage
- **THEN** the generated low-level command includes explicit `--input` and `--output` arguments derived from the manifest

#### Scenario: Low-level CLI remains usable

- **WHEN** users run existing low-level scripts directly with documented arguments
- **THEN** those scripts continue to accept their current inputs and outputs without requiring a manifest

### Requirement: Status and stale artifact detection

The system SHALL provide status inspection for an experiment and SHALL distinguish missing, complete, and stale or mismatched stage outputs when enough provenance metadata is available.

#### Scenario: Report missing outputs

- **WHEN** an expected stage output does not exist
- **THEN** the status command reports that stage as missing or not started

#### Scenario: Report completed outputs

- **WHEN** expected outputs and matching provenance metadata exist for a stage
- **THEN** the status command reports that stage as complete

#### Scenario: Report stale outputs

- **WHEN** an expected output exists but its recorded method, split, input path, or effective config does not match the manifest
- **THEN** the status command reports that output as stale or mismatched instead of silently reusing it

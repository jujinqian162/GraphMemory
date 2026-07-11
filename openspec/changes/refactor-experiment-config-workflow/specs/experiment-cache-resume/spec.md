## ADDED Requirements

### Requirement: Run state is typed and minimal
The system SHALL persist a Pydantic `RunState` containing run identity and mode, resolved-config and override paths, selected methods and stages, artifact layout references, MLflow parent run id, creation time, and schema version. It MUST NOT duplicate live stage status, broad counts, timings, environment, or notes.

#### Scenario: New run state
- **WHEN** a valid name is initialized by plan or run
- **THEN** `RunState` SHALL be written atomically and SHALL validate when read back

#### Scenario: Live status inspection
- **WHEN** status is requested
- **THEN** stage status SHALL be derived from current artifacts and summaries rather than copied from `RunState`

### Requirement: Every primary artifact has a typed stage summary
Each prepare, graph, pair, tune, train, retrieve, evaluate, and aggregate primary artifact SHALL have an adjacent Pydantic `StageRunSummary` recording identity, terminal status, timestamps, effective config, declared inputs and outputs, relevant bound artifacts, observations, and optional MLflow child id.

#### Scenario: Successful stage
- **WHEN** a stage finishes and all declared outputs validate
- **THEN** the shared lifecycle SHALL atomically write a successful summary adjacent to the primary artifact

#### Scenario: Failed stage
- **WHEN** a stage raises an exception or tracking write fails
- **THEN** the lifecycle SHALL write a failed summary when possible and re-raise the error

### Requirement: Artifact status has four explicit states
Status inspection SHALL return `missing`, `complete`, `stale`, or `alias` by comparing the typed expected invocation, artifact kind and validity, and adjacent summary. Artifact existence alone or MLflow run status MUST NOT establish completeness.

#### Scenario: Missing summary
- **WHEN** an output exists but its adjacent stage summary is absent
- **THEN** status SHALL be `stale`

#### Scenario: Mismatched effective config
- **WHEN** an output and successful summary exist but the summary's effective config differs from the invocation
- **THEN** status SHALL be `stale`

#### Scenario: Valid directory checkpoint
- **WHEN** a Dense-FT model directory and matching successful summary satisfy the declared directory artifact contract
- **THEN** status SHALL be `complete`

#### Scenario: Baseline alias
- **WHEN** an ablation output is explicitly bound to a valid baseline artifact alias
- **THEN** status SHALL be `alias`

### Requirement: Resume prunes only a completed prefix
With cache enabled, the resume planner SHALL remove only the longest continuous prefix of `complete` or `alias` invocations from the selected ordered plan and SHALL retain the first `missing` or `stale` invocation plus every later invocation.

#### Scenario: Interrupted run
- **WHEN** the first three invocations are complete, the fourth is stale, and a later invocation appears complete
- **THEN** resume SHALL skip only the first three and SHALL include the fourth and every later invocation

#### Scenario: Cache disabled
- **WHEN** `cache.enabled=false`
- **THEN** no completed-prefix pruning SHALL occur and executed stages SHALL still write ordinary typed summaries

### Requirement: Cache scope is intentionally limited
The cache implementation MUST NOT add artifact hashes, source fingerprints, content digests, cross-run reuse, or independent DAG-node cache selection. Code/raw changes at stable paths SHALL be handled by a new name, explicit reset, or cache-disabled rerun.

#### Scenario: Separate named runs
- **WHEN** two different names produce equivalent configs and artifacts
- **THEN** neither run SHALL consume the other run's cache entries

#### Scenario: Later complete node after stale node
- **WHEN** a stale node precedes a complete node in the ordered plan
- **THEN** the complete later node SHALL still be rerun because cache behavior is ordered-prefix only

### Requirement: Stage-specific validation is retained
Status inspection SHALL preserve stage-specific expected input, output, selected-config, checkpoint, split, method, variant, and effective-config comparisons for every supported stage and artifact shape.

#### Scenario: Selected config mismatch
- **WHEN** graph-rerank retrieval's summary references a different selected tuning config than the invocation
- **THEN** retrieval status SHALL be `stale`

#### Scenario: Failed residual output
- **WHEN** a failed stage leaves a primary output path behind
- **THEN** the output SHALL remain `stale` and SHALL NOT be pruned as a cache hit


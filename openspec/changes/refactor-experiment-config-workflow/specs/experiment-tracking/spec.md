## ADDED Requirements

### Requirement: MLflow uses one repository-wide SQLite store
Orchestrated experiments SHALL use one configured SQLite backend store and one artifact root so runs are comparable across names and multirun jobs. Tracking initialization failure MUST abort the experiment and MUST NOT silently disable tracking or select another backend.

#### Scenario: Backend unavailable
- **WHEN** the configured SQLite store cannot be initialized or written
- **THEN** run SHALL fail before executing untracked scientific stages

#### Scenario: Cross-run comparison
- **WHEN** two named experiments complete
- **THEN** both parent runs SHALL be queryable from the same MLflow tracking store

### Requirement: Hydra jobs and stage attempts map to parent and child runs
Each Hydra single-run or multirun job SHALL map to one parent MLflow run. Each actually executed stage/method/split/variant attempt SHALL map to one nested child run. Cache hits MUST NOT create child runs.

#### Scenario: Exact resume
- **WHEN** an identical named run resumes after interruption
- **THEN** it SHALL reuse the persisted parent run id and create a new child only for each newly executed attempt

#### Scenario: Pruned stage
- **WHEN** an invocation is pruned as a complete cache prefix
- **THEN** tracking SHALL not record a new successful child attempt for that invocation

### Requirement: Tracking records scientific and provenance metadata
The parent and child runs SHALL record flattened scientific parameters, dataset/profile/method selections, seed, device, top-k, source revision and dirty flag, Python and dependency versions, stage counts and timings, train epoch metrics with step, tuning best values, retrieval/evaluation/aggregate metrics, status, and exceptions as applicable.

#### Scenario: Training epochs
- **WHEN** a training stage emits epoch metrics
- **THEN** MLflow SHALL log each metric using the epoch as step while the local training metric JSONL remains authoritative

#### Scenario: Tuning candidates
- **WHEN** tuning evaluates multiple candidates
- **THEN** tracking SHALL upload one candidate table, log best-candidate parameters and metrics, and SHALL not create one child run per candidate

### Requirement: Artifact uploads are deny-by-default and allowlisted
The tracking adapter SHALL upload only resolved configs, override lists, stage summaries, selected tuning configs, candidate tables, aggregate CSVs, report images, and small failure summaries. Raw/processed datasets, graphs, train pairs, full predictions, checkpoints, and model directories MUST NOT be uploaded.

#### Scenario: Large graph artifact
- **WHEN** a graph stage completes
- **THEN** MLflow SHALL record graph path, size, kind, and role metadata without copying the graph artifact

#### Scenario: Aggregate CSV
- **WHEN** aggregate results are produced
- **THEN** the allowlist SHALL permit the final CSVs to be logged as curated artifacts

### Requirement: Tracking failures preserve cache correctness
Any MLflow initialization, parameter, metric, tag, or allowlisted artifact logging failure during an orchestrated stage SHALL fail the stage attempt. The local summary SHALL not claim success and output existence MUST remain stale.

#### Scenario: Metric logging fails after output write
- **WHEN** a stage writes its primary output but MLflow metric logging then raises
- **THEN** the stage SHALL exit nonzero and its summary/status SHALL be failed or stale

### Requirement: Direct scripts do not create orphan runs
Low-level stage scripts invoked without an explicit parent tracking context SHALL complete their local artifact and summary responsibilities without creating a standalone MLflow run.

#### Scenario: Developer direct invocation
- **WHEN** a developer runs `scripts/run_retrieval.py --config <stage.yaml>` outside experiment run
- **THEN** retrieval SHALL write local outputs and a summary while MLflow receives no new parent or child run


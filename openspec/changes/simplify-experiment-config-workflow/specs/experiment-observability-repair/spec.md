## ADDED Requirements

### Requirement: MLflow uses the locked repository-wide paths
The repository SHALL use `runs/.mlflow/tracking.db` as the SQLite backend and `runs/.mlflow/artifacts/` as the managed artifact root. No named run SHALL create its own store, and `mlruns/` MUST NOT be the active path.

#### Scenario: First tracked run
- **WHEN** the first experiment initializes tracking
- **THEN** the SQLite database and managed artifact directory SHALL be created below `runs/.mlflow/`

#### Scenario: Cross-run comparison
- **WHEN** multiple names or multirun jobs execute
- **THEN** their parent and child runs SHALL be recorded in the same repository-wide store

### Requirement: Tracking remains a strict mirror
One Hydra job SHALL map to one parent run, and each executed stage attempt SHALL map to one child run. Cache hits SHALL create no child run. Tracking failures SHALL fail the attempt and leave local output stale; no silent disable, file-store fallback, or alternate backend is allowed.

#### Scenario: Exact resume
- **WHEN** an identical named run resumes
- **THEN** it SHALL reuse the persisted parent id and create children only for newly executed attempts

#### Scenario: Tracking failure
- **WHEN** a metric, parameter, artifact, or termination write fails
- **THEN** the stage summary SHALL be failed or stale and the execution error SHALL propagate

### Requirement: Local scientific artifacts remain authoritative
Predictions, checkpoints/model directories, train pairs, complete datasets, graphs, local metric files, failure cases, and aggregate tables SHALL remain usable without MLflow access. Large artifacts SHALL be recorded as path/size/kind/role metadata and MUST NOT be uploaded by default.

#### Scenario: Offline delivery
- **WHEN** run artifacts are collected without access to the MLflow store
- **THEN** local outputs SHALL be sufficient to validate and deliver the scientific result

### Requirement: Evaluation and aggregate results are queryable metrics
Tracking SHALL log stable numeric metrics for evaluation rows and multi-row aggregate result tables while also uploading allowed aggregate tables as curated artifacts. It MUST NOT skip a result table merely because it contains more than one method row.

#### Scenario: Seven-method aggregate
- **WHEN** the default seven-method workflow completes aggregation
- **THEN** MLflow SHALL contain queryable numeric result metrics with stable method/row identity and the main, path, and efficiency tables as curated artifacts

#### Scenario: Ablation aggregate
- **WHEN** ablation aggregation completes
- **THEN** variant metrics SHALL be queryable by typed method and variant identity and the ablation table SHALL remain a curated artifact

### Requirement: Tracking configuration has one owner
The canonical root config SHALL own the fixed database path, artifact root, and experiment name. A one-option tracking config group and duplicated tracking defaults MUST NOT remain.

#### Scenario: Resolved configuration
- **WHEN** the canonical experiment config is resolved
- **THEN** it SHALL contain the fixed absolute tracking paths without composing `tracking/default.yaml`

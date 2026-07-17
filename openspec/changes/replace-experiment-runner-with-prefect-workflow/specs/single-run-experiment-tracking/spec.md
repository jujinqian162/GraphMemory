## ADDED Requirements

### Requirement: One final method maps to one MLflow run
Each Hydra job and Prefect Flow run SHALL create exactly one top-level MLflow run for its final method/variant. The tracking layer MUST NOT create parent summary runs, baseline child runs, dependency child runs, or stage child runs.

#### Scenario: Stateless experiment
- **WHEN** a BM25 Flow completes
- **THEN** MLflow contains one run carrying the BM25 configuration and final result

#### Scenario: Composite trainable experiment
- **WHEN** Dense-FT executes only as a dependency of Dense-FT-seeded R-GCN
- **THEN** both training stages are represented in one final R-GCN MLflow run without a separate Dense-FT child run

### Requirement: Active tracking uses the MLflow fluent API only
The experiment entrypoint SHALL start, log to, and finish the sole active MLflow run through MLflow's fluent API. Active tracking MUST NOT instantiate `MlflowClient`, query for a reusable run ID, mutate a non-active run, or create parent/child relationships. A separate read-only reporting boundary MAY use `mlflow.search_runs` to compare completed peer runs, but MUST NOT write experiment state.

#### Scenario: Log and finish a successful experiment
- **WHEN** one final method completes
- **THEN** fluent active-run calls log all parameters, metrics, tags, and small artifacts and finish that same run without constructing an `MlflowClient`

#### Scenario: Fail the active experiment
- **WHEN** experiment execution or required tracking fails after the run starts
- **THEN** the entrypoint marks the active run failed through `mlflow.end_run(status="FAILED")` or equivalent fluent context handling without a client or run-ID lookup

#### Scenario: Compare completed peer runs
- **WHEN** a reporting utility needs an ablation table
- **THEN** it may read peer runs through `mlflow.search_runs` outside the Flow but cannot update those runs or turn them into parent/child runs

### Requirement: Current runs are complete on cache hits
Experiment-level MLflow logging SHALL occur in the Flow after Task results are available. A current MLflow run SHALL receive resolved configuration, final method and variant identity, final metrics, processed asset references, and applicable small reports whether its Tasks executed or returned cached results. Prefect remains the Task-state and cache-hit authority; MLflow SHALL NOT duplicate per-Task state records.

#### Scenario: Fully cached experiment
- **WHEN** all scientific Tasks return cached results under a new run name
- **THEN** the new MLflow run still contains complete comparable configuration, final metrics, and asset references

### Requirement: Final metric keys are comparable across methods
All final-method runs SHALL log applicable numeric outcomes under the same method-independent `final.*` metric keys. Method, variant, dataset, profile, and seed SHALL be parameters or tags rather than metric-key prefixes. Non-numeric or inapplicable values MUST NOT be fabricated as numeric metrics.

#### Scenario: Compare BM25 and R-GCN
- **WHEN** the two runs are selected in MLflow Compare Runs
- **THEN** shared metrics such as recall, full support, and path recall align under identical keys

### Requirement: Processed assets are referenced without large MLflow duplication
MLflow SHALL log URI, digest, kind, origin, and size metadata for large processed assets and MAY upload their small manifests. It MUST NOT upload duplicate copies of prepared datasets, graphs, training pairs, predictions, checkpoints, model directories, or optimizer state merely to make the run self-contained.

#### Scenario: Log a Dense-FT model
- **WHEN** a Dense-FT model is freshly trained or reused from cache
- **THEN** MLflow records its processed model reference and small manifest without copying the full model directory into the MLflow artifact store

### Requirement: Training history is distinguished from current execution
Small training curves and summaries associated with a model MAY be projected into the current final run, and the run SHALL identify their processed asset origin. Historical training duration or resource use MUST NOT be labelled as work performed by the current Flow.

#### Scenario: Cached trained model
- **WHEN** a current experiment reuses a cached R-GCN checkpoint and training history
- **THEN** MLflow may show the model's epoch curve with its processed asset origin but omits current training duration rather than presenting historical duration as current work

### Requirement: Current runtime metrics require fresh measurement
Timing stored on a cached artifact SHALL be treated as origin provenance and MUST NOT be logged as current-run runtime or benchmark performance. Comparable current efficiency metrics SHALL come only from a fresh non-cached benchmark Task with the configured warmup and repetition protocol.

#### Scenario: Cached rankings without benchmark
- **WHEN** retrieval returns cached rankings and benchmark mode is not requested
- **THEN** the run logs quality metrics and cache provenance but omits current retrieval latency rather than reusing the artifact's old latency

#### Scenario: Fresh benchmark
- **WHEN** benchmark mode is requested after cached scientific inputs and models are resolved
- **THEN** a non-cached benchmark Task runs and its fresh measurements are logged under the designated benchmark or final efficiency keys

### Requirement: MLflow does not influence cache correctness
MLflow run existence, status, tags, parameters, metrics, and artifacts MUST NOT participate in Task cache keys or processed artifact resolution. Deleting or changing an MLflow record MUST NOT invalidate an otherwise valid Prefect cache result.

#### Scenario: Tracking database unavailable before computation
- **WHEN** tracking initialization cannot establish the required MLflow run
- **THEN** the Flow fails explicitly without mutating Task cache identity or substituting MLflow as artifact storage

### Requirement: Tracking failures propagate with consistent local state
The tracking layer SHALL mark the single MLflow run failed when experiment execution or required tracking fails. Valid processed assets produced before the failure SHALL remain reusable, and output-only run files SHALL never become fallback cache truth.

#### Scenario: Evaluation logging fails
- **WHEN** final MLflow metric logging raises an error after evaluation produced a valid processed result
- **THEN** the Flow and MLflow run fail, the processed evaluation asset remains valid, and a rerun may reuse it before retrying current-run tracking

### Requirement: Study identity groups independent jobs
Hydra multirun jobs SHALL carry a stable user-visible study or run-name tag that allows independent single-method MLflow runs to be filtered together. Group identity MUST NOT create an MLflow parent run or affect scientific Task cache identity.

#### Scenario: Filter an ablation study
- **WHEN** a user filters MLflow by the shared ablation study identity
- **THEN** all variant runs appear as peer runs with comparable metric keys

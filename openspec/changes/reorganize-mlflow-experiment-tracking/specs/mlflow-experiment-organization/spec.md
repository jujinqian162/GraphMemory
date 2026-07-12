## ADDED Requirements

### Requirement: Multirun jobs have concise identities
The system SHALL place each Hydra multirun job below `runs/<name>/` using `<job-number>_<varying-leaf-key>=<value>` for a single varying dimension, and SHALL include each varying leaf-key assignment in deterministic order when more than one dimension varies. Fixed command overrides such as dataset, profile, and method selection MUST NOT appear in every job leaf. The complete resolved configuration and override list SHALL remain persisted inside the job directory.

#### Scenario: Single-dimension R-GCN layer sweep
- **WHEN** `name=rgcn-num-layer` sweeps `method_configs.dense_rgcn_graph_retriever.train.model.num_layers=2,3,4` while dataset, profile, and methods remain fixed
- **THEN** the job directories SHALL be `0_num_layers=2`, `1_num_layers=3`, and `2_num_layers=4`

#### Scenario: Complete overrides remain available
- **WHEN** a concise multirun leaf omits fixed overrides from its name
- **THEN** `config/resolved.yaml`, `config/overrides.yaml`, and the multirun manifest SHALL retain the full values required to reproduce that job

#### Scenario: Ambiguous leaf keys
- **WHEN** multiple varying override paths collapse to the same leaf key
- **THEN** planning SHALL fail with an actionable ambiguity error rather than silently producing a colliding or misleading directory name

### Requirement: A parent run is an experiment summary container
Each Hydra single-run or multirun job SHALL own one parent MLflow run. The parent SHALL publish only concise experiment identity and selection parameters, a readable all-baseline final-results table in its Overview description, and organized job-level artifacts. The parent MUST NOT publish method-specific configuration trees, native result metrics, generated comparison plots, or model-training series.

#### Scenario: Concise parent parameters
- **WHEN** a parent run is created
- **THEN** its parameters SHALL include the run name, dataset, profile, seed, device, selected methods, top-k, stage bounds, cache and ablation selections, and applicable multirun job identity
- **AND** no parent parameter key SHALL descend from `method_configs` or a method search-space configuration

#### Scenario: Completed full experiment summary
- **WHEN** aggregation succeeds for a job containing multiple baselines
- **THEN** the parent Overview description SHALL contain one readable table with one row per baseline and all final result columns produced by the aggregate contract

#### Scenario: Parent artifacts
- **WHEN** a job completes or fails after producing tracking-eligible files
- **THEN** the parent Artifacts view SHALL organize the resolved configuration and overrides under `config/`, aggregate result tables under `results/`, and shared stage status and summaries under `workflow/`

#### Scenario: Parent has no comparison data
- **WHEN** a parent run is inspected
- **THEN** it SHALL contain no final evaluation metric keys, epoch-series metric keys, repository-generated comparison image, or repository-generated comparison chart configuration

#### Scenario: Partial run without aggregate results
- **WHEN** a stage-bounded run completes without producing aggregate result tables
- **THEN** the parent SHALL present the concise experiment summary and available job artifacts without fabricating an all-baseline result table

### Requirement: Each user-visible baseline owns one child run
The system SHALL create one MLflow child run for each user-visible selected baseline identity, where an ablation variant forms a distinct baseline identity. Shared dataset preparation, graph construction, and aggregate stages MUST NOT create MLflow children. Pair generation, tuning, training, retrieval, and evaluation observations SHALL be attached to the owning baseline child without changing the local stage execution order.

#### Scenario: Seven selected baselines
- **WHEN** a successful experiment selects seven ordinary baselines
- **THEN** its parent SHALL own exactly seven baseline child runs and no prepare, graph, pair, tune, train, retrieve, evaluate, or aggregate stage child runs

#### Scenario: Trainable baseline lifecycle
- **WHEN** a trainable baseline executes pair, train, retrieve, and evaluate stages
- **THEN** all four stages SHALL contribute parameters, metrics, tags, and artifacts to the same baseline child

#### Scenario: Non-trainable baseline lifecycle
- **WHEN** a non-trainable baseline executes retrieval and evaluation
- **THEN** those observations SHALL be attached to one baseline child without a fabricated training series

#### Scenario: Hidden training dependency
- **WHEN** a selected baseline requires a hidden trainable dependency that is not itself user-selected
- **THEN** the dependency SHALL NOT create an additional baseline child and its local summaries and artifact references SHALL be organized below the selected baseline child's dependency artifacts

#### Scenario: Explicitly selected dependency
- **WHEN** the dependency method is also present in the user-visible method selection
- **THEN** it SHALL own its own baseline child and the dependent baseline SHALL reference rather than duplicate its tracked training observations

#### Scenario: Baseline failure
- **WHEN** a method-owned stage fails
- **THEN** the owning baseline child and parent SHALL terminate as failed and the child SHALL identify the failed local stage through tags and artifacts

#### Scenario: Shared-stage failure
- **WHEN** a shared prepare or graph stage fails before any baseline-owned stage executes
- **THEN** the parent SHALL terminate as failed without creating synthetic failed children for baselines that never started

### Requirement: Baseline children expose comparable final metrics
Each evaluated baseline child SHALL log each applicable final numeric result exactly once using the same method-independent MLflow metric key across methods, datasets, single runs, and multirun jobs. Method identity SHALL live in the child run name, parameters, and tags rather than in the metric key. Non-numeric `N/A` results SHALL remain visible in result artifacts and MUST NOT be converted into numeric metric values.

#### Scenario: Common metric keys
- **WHEN** BM25, Dense, and Dense-FT children each produce `Recall@10`
- **THEN** every child SHALL log that result under `final.recall_at_10` rather than a method-prefixed or stage-prefixed key

#### Scenario: Complete final metric mapping
- **WHEN** a final evaluation row contains supported numeric main, path, connectivity, or efficiency results
- **THEN** tracking SHALL map them to stable `final.*` keys including recall, evidence F1, full support, MRR, connected-evidence recall, query-evidence connectivity, applicable path and edge recall, retrieval latency, index and graph construction time, memory size, and retrieved node and edge counts

#### Scenario: Dataset-inapplicable result
- **WHEN** `Path Recall@10` or `Edge Recall@10` is `N/A` for a dataset
- **THEN** the baseline result artifact and parent result table SHALL retain `N/A` and the child SHALL omit the corresponding numeric MLflow metric

#### Scenario: Compare runs
- **WHEN** users select baseline children from different methods or multirun jobs in MLflow Compare Runs
- **THEN** their common `final.*` keys SHALL be directly comparable without selecting method-specific metric names

### Requirement: Trainable baseline children expose epoch series
Each trainable baseline child SHALL log the numeric epoch records from its authoritative local training metric file using the epoch as the MLflow step. Training-series keys SHALL be stable across trainable methods where the local record has the same meaning.

#### Scenario: R-GCN training curves
- **WHEN** an R-GCN baseline records multiple epochs
- **THEN** its child SHALL expose `train.train_loss`, `train.dev_loss`, `train.dev_recall_at_5`, `train.dev_full_support_at_5`, `train.dev_full_support_at_10`, `train.dev_mrr`, `train.learning_rate`, and `train.grad_norm` as applicable epoch-indexed series

#### Scenario: Dense-FT training curves
- **WHEN** a Dense-FT baseline records epoch metrics
- **THEN** its child SHALL expose `train.train_loss`, its selected development metric, best development metric, best epoch, and global step as applicable using the recorded epoch step

#### Scenario: Nested non-numeric training observation
- **WHEN** an epoch record contains a mapping, list, boolean, string, null, NaN, or infinite value
- **THEN** tracking SHALL not log that value as an MLflow metric and SHALL preserve it only in the local or uploaded training artifact as allowed

### Requirement: Parameters and artifacts follow baseline ownership
Each baseline child SHALL publish the selected method's configuration under concise method-relative parameter keys and SHALL organize its eligible tuning, training, retrieval, evaluation, dependency, and stage-summary files by responsibility. Large scientific artifacts SHALL remain local and be represented only by path, kind, size, and role metadata.

#### Scenario: Search method-specific epochs
- **WHEN** a user opens an R-GCN baseline child and searches its parameters for `epochs`
- **THEN** the child SHALL expose the effective trainer epoch value without requiring a parent-level `method_configs.<method>` prefix

#### Scenario: Baseline artifact organization
- **WHEN** a baseline produces eligible files
- **THEN** its Artifacts view SHALL organize method configuration under `config/`, selected tuning outputs under `tuning/`, training metrics and summaries under `training/`, evaluation metrics under `evaluation/`, and method stage summaries under `workflow/`

#### Scenario: Large artifact policy
- **WHEN** a baseline produces datasets, graphs, train pairs, predictions, checkpoints, or model directories
- **THEN** MLflow SHALL record their path metadata without uploading or duplicating those artifacts

### Requirement: Metric logging avoids duplicate and operational pseudo-metrics
Tracking SHALL log final baseline metrics once on the baseline child and epoch series once on trainable baseline children. It MUST NOT mirror the same evaluation values again from aggregate tables, and it MUST NOT log stage counts, stage timings, artifact sizes, or every numeric aggregate cell as model metrics. Those observations SHALL remain available through parameters, tags, Overview content, stage summaries, or artifacts as appropriate.

#### Scenario: Evaluation followed by aggregation
- **WHEN** a baseline evaluation is later included in the parent aggregate tables
- **THEN** the baseline child SHALL retain one copy of each `final.*` metric and aggregation SHALL not create another metric-bearing run or duplicate metric key

#### Scenario: Operational observations
- **WHEN** a stage summary contains counts and timings
- **THEN** tracking SHALL expose them outside the native model-metric namespace and Model metrics SHALL remain limited to final comparison metrics and genuine training series

#### Scenario: Native scalar rendering limitation
- **WHEN** MLflow renders a baseline child's native final scalar metrics as single-point charts
- **THEN** the repository SHALL accept that standard UI behavior because those metrics are required for Compare Runs and SHALL NOT add generated comparison plots or custom MLflow frontend code to mask it

### Requirement: Baseline children remain stable across cache and resume
A successful full job SHALL expose one child per selected baseline regardless of whether its local stages executed or were satisfied from valid cache. Exact resume SHALL reuse the existing parent and baseline children rather than creating duplicates. MLflow state MUST remain observability only and MUST NOT participate in local cache validity decisions.

#### Scenario: Fully cached baseline
- **WHEN** every local stage for a selected baseline is a valid cache hit
- **THEN** the job SHALL still expose one baseline child populated from the authoritative local configuration, summaries, metrics, and eligible artifacts

#### Scenario: Exact resume
- **WHEN** an identical named job resumes after an interruption
- **THEN** tracking SHALL reuse the persisted parent and the unique existing child for each baseline identity and append only newly available stage observations

#### Scenario: Cache decision
- **WHEN** MLflow reports a child as finished but local artifact and summary validation reports missing or stale state
- **THEN** execution SHALL follow the local validation result and MUST NOT treat MLflow status as cache truth

### Requirement: The replacement has no compatibility surface
The implementation SHALL adopt the new directory and tracking organization directly. It MUST NOT add a tracking schema/version tag, compatibility reader, dual-write path, legacy child projection, old-run migration, database rewrite, or automatic conversion of existing run directories.

#### Scenario: New execution after cutover
- **WHEN** a new run name is executed after the change
- **THEN** only the concise multirun and parent/baseline organization SHALL be produced

#### Scenario: Existing tracking data
- **WHEN** the shared SQLite store already contains runs produced by the former stage-child organization
- **THEN** those records SHALL remain untouched historical records and the implementation SHALL not tag, rewrite, migrate, or duplicate them

#### Scenario: Reusing an old named run
- **WHEN** a user wants to execute under a name whose local state was created by the former organization
- **THEN** operations documentation SHALL require a new name or an explicit normal reset before execution, with no automatic compatibility behavior

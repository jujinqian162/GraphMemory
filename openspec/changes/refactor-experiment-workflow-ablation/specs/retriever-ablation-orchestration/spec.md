## ADDED Requirements

### Requirement: Config-controlled ablation expansion
The experiment runner SHALL support an `enable_ablation` experiment-config switch that expands registered ablation variants for selected methods with registered suites.

#### Scenario: Ablation disabled preserves ordinary run units
- **WHEN** an experiment config sets `enable_ablation` to `false` or omits it
- **THEN** the runner SHALL plan ordinary selected-method run units only
- **THEN** the runner SHALL preserve non-ablation artifact paths and command behavior

#### Scenario: Ablation enabled expands registered R-GCN suite
- **WHEN** an experiment config sets `enable_ablation` to `true`
- **WHEN** method `dense_rgcn_graph_retriever` is selected
- **THEN** the planner SHALL expand the registered R-GCN ablation suite
- **THEN** plan output SHALL include variant-qualified commands for non-baseline variants

#### Scenario: Config subset narrows expanded variants
- **WHEN** an ablation-enabled experiment config lists an ordered variant subset for `dense_rgcn_graph_retriever`
- **THEN** the planner SHALL expand only that subset plus the aliased `full_rgcn` baseline row

#### Scenario: Unknown configured variant fails fast
- **WHEN** experiment config requests a variant not registered for the selected method
- **THEN** initialization SHALL fail before low-level command execution
- **THEN** the error SHALL list the suite's allowed variant values

### Requirement: R-GCN ablation suite coverage
The registered R-GCN ablation suite SHALL expose the baseline, graph-view variants, model-structure variants, and pair-sampling variant required for the first automated ablation table.

#### Scenario: Discover registered R-GCN variants
- **WHEN** a user lists ablations for `dense_rgcn_graph_retriever`
- **THEN** the output SHALL include `full_rgcn`, `wo_bridge`, `wo_entity_overlap`, `wo_sequential`, `wo_query_overlap`, `wo_graph`, `wo_edge_type`, `wo_edge_weight`, `wo_seed_score`, and `wo_hard_negatives`

#### Scenario: Random-edge variant remains absent
- **WHEN** a user lists ablations for `dense_rgcn_graph_retriever`
- **THEN** the output SHALL NOT claim support for a `random_edges` variant

### Requirement: Main-config inheritance with minimal overrides
Each non-baseline R-GCN variant SHALL inherit the resolved main R-GCN training config and SHALL apply only its registered minimal override.

#### Scenario: Model-only override preserves training hyperparameters
- **WHEN** the planner resolves variant `wo_graph`
- **THEN** the generated variant config SHALL preserve the main resolved optimization and pair-sampling sections
- **THEN** the generated variant config SHALL apply the registered model override for graph-propagation removal

#### Scenario: Edge-view override preserves source graph artifacts
- **WHEN** the planner resolves variant `wo_entity_overlap`
- **THEN** the generated variant config SHALL preserve the main graph artifact paths and optimization section
- **THEN** the generated variant config SHALL apply the registered model-visible edge override

#### Scenario: Hard-negative override changes pair sampling only
- **WHEN** the planner resolves variant `wo_hard_negatives`
- **THEN** the generated variant config SHALL disable registered hard-negative sampling fields
- **THEN** the generated variant config SHALL preserve the main model and optimization sections

#### Scenario: Generated configs are persisted for audit
- **WHEN** a non-baseline variant is initialized
- **THEN** its effective training config SHALL be written under `runs/<experiment>/ablations/<method>/<variant>/effective_training_config.json`

### Requirement: Baseline alias instead of duplicate training
The R-GCN suite's `full_rgcn` variant SHALL alias the ordinary main R-GCN run rather than scheduling an identical second run.

#### Scenario: Full baseline reuses main artifacts
- **WHEN** the planner expands `full_rgcn`
- **THEN** the baseline row SHALL reference the main R-GCN checkpoint, prediction, and metric artifacts
- **THEN** the planner SHALL NOT schedule duplicate `pairs`, `train`, `retrieve`, or `evaluate` commands for `full_rgcn`

### Requirement: Variant-local artifact isolation
Each non-baseline ablation variant SHALL write invalidated and downstream artifacts under its own deterministic namespace.

#### Scenario: Structural variant paths are isolated
- **WHEN** the planner expands `wo_edge_type`
- **THEN** its checkpoint, prediction, metrics, debug output, run summaries, and effective training config SHALL be written below `runs/<experiment>/ablations/dense_rgcn_graph_retriever/wo_edge_type/`

#### Scenario: Pair-sampling variant paths include pairs
- **WHEN** the planner expands `wo_hard_negatives`
- **THEN** its train-pair artifact and pair summaries SHALL be written below `runs/<experiment>/ablations/dense_rgcn_graph_retriever/wo_hard_negatives/`

### Requirement: Variant CLI filtering and discovery
The experiment runner SHALL provide CLI controls for inspecting and narrowing ablation work without requiring users to edit orchestration code.

#### Scenario: List suites and variants
- **WHEN** a user runs `scripts/experiment.py ablations list`
- **THEN** the output SHALL list methods with registered suites and their allowed variant values

#### Scenario: List variants for one method
- **WHEN** a user runs `scripts/experiment.py ablations list --method dense_rgcn_graph_retriever`
- **THEN** the output SHALL list the R-GCN suite baseline and variants with their changed dimensions

#### Scenario: Plan selected variants only
- **WHEN** a user supplies repeated `--variant` filters while planning or running an ablation-enabled experiment
- **THEN** the planner SHALL schedule only those non-baseline variants plus required shared prerequisites

#### Scenario: Plan ablation work only
- **WHEN** a user supplies `--ablations-only`
- **THEN** the planner SHALL omit ordinary selected-method commands except artifacts required as shared prerequisites or aliases for selected ablation variants

### Requirement: Ablation-aware status inspection
The experiment runner SHALL report status for each expanded ablation variant and SHALL identify aliased, missing, complete, and stale artifacts where evidence is available.

#### Scenario: Status includes each variant
- **WHEN** a user inspects an ablation-enabled experiment
- **THEN** status output SHALL include method and variant qualifiers for variant-local stages

#### Scenario: Baseline status shows alias
- **WHEN** a user inspects the `full_rgcn` ablation baseline
- **THEN** status output SHALL identify that its artifacts alias the ordinary main R-GCN run

### Requirement: Write ablation result table
The aggregate stage SHALL write `runs/<experiment>/tables/ablation_results.csv` for an ablation-enabled run using an explicit run-local metric index.

#### Scenario: Aggregate indexed variant metrics
- **WHEN** ablation evaluation metrics exist and the aggregate stage runs
- **THEN** the runner SHALL supply a run-local ablation metric index to `scripts/aggregate_tables.py`
- **THEN** the aggregate script SHALL write one table row per indexed baseline or variant metric file

#### Scenario: Ablation table columns are stable
- **WHEN** `ablation_results.csv` is written
- **THEN** the table SHALL include `Method`, `Variant`, `Recall@5`, `Full Support@5`, `Connected Evidence Recall@10`, `Path Recall@10`, and `Retrieval Latency / Query`

#### Scenario: Non-ablation aggregation remains compatible
- **WHEN** ablation is disabled
- **THEN** `scripts/aggregate_tables.py` SHALL continue to write existing main, path, and efficiency tables without requiring ablation arguments

### Requirement: Edge-view prediction artifacts reflect visible edges
R-GCN edge-view variants SHALL write retrieved-subgraph edges that match the model-visible graph view while preserving existing reference-graph connectivity metric semantics.

#### Scenario: Removed edge type is absent from prediction subgraph
- **WHEN** variant `wo_bridge` retrieves ranked nodes
- **THEN** its prediction artifact SHALL NOT include `bridge` edges in `retrieved_subgraph.edges`

#### Scenario: Reference connectivity metric remains comparable
- **WHEN** an edge-view variant is evaluated
- **THEN** existing `Connected Evidence Recall@10` SHALL continue to use the reference graph input
- **THEN** documentation SHALL identify that reference-graph meaning

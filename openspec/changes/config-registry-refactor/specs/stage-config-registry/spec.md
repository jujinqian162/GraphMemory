## ADDED Requirements

### Requirement: Single stage config loader entrypoint
The system SHALL expose one public loader entrypoint that accepts a stage config spec and argv, and SHALL keep config-file reading, profile patching, registry patching, CLI override application, typed structuring, and resolved-config serialization inside the config layer.

#### Scenario: Loader applies fixed precedence
- **WHEN** a stage config is loaded from a base config, selected profile, registry patch, and CLI overrides
- **THEN** the resolved config applies base values first, profile values second, registry patch values third, and CLI override values last

#### Scenario: Loader rejects unknown config fields
- **WHEN** a config file or patch contains fields that are not part of the selected stage root config
- **THEN** loading fails before stage execution starts

#### Scenario: Loader does not expose intermediate source APIs
- **WHEN** callers import the config loader public API
- **THEN** they can call `load(spec, argv)`, `to_json(config)`, and `write_resolved(path, config)`
- **THEN** they do not need public `ConfigSource`, `load_cli_config`, or `load_profiled_file` APIs

### Requirement: Scripts remain thin stage adapters
The system SHALL keep scripts responsible for stage selection, artifact IO, artifact validation, fixed stage runner or domain service invocation, output writing, run summaries, logging, and exit codes, while config parsing and method dispatch are owned by config and registry modules.

#### Scenario: Retrieval script selects a stage config
- **WHEN** `scripts/run_retrieval.py` starts
- **THEN** it loads `Registry.configs.RETRIEVE` through `ConfigLoader.load`
- **THEN** it does not construct dense runtime objects, checkpoint runtime objects, or method-family builders directly

#### Scenario: Scripts preserve public parser contracts
- **WHEN** parser contract tests inspect retrieval, pair-build, train, evaluate, and experiment scripts
- **THEN** public CLI flags, choices, defaults, and compatibility helper entrypoints remain available unless a later OpenSpec change explicitly changes them

### Requirement: Registry owns method and stage metadata
The system SHALL make `graph_memory.registry` the source of truth for stage config specs, public method ids, method capability metadata, method settings unions, builder maps, workflow-facing projections, and ablation patches.

#### Scenario: Retrieval catalog is a projection
- **WHEN** callers import `graph_memory.retrieval.catalog` or `graph_memory.retrieval_registry`
- **THEN** legacy metadata APIs remain available
- **THEN** their values are projected from `graph_memory.registry` rather than an independent retrieval catalog table

#### Scenario: Retrieval builder dispatch uses settings type
- **WHEN** a retrieve stage runner builds a retrieval method
- **THEN** it calls `Registry.retrieval.build(config.job, deps)`
- **THEN** runtime construction dispatch is selected by the concrete settings type rather than a public method string or `builder_id`

#### Scenario: Compatibility builder ids are not runtime source of truth
- **WHEN** legacy `builder_id` values exist for docs, workflow compatibility, or old callers
- **THEN** they are generated projection metadata
- **THEN** new stage execution does not use `builder_id` as the dispatch input

### Requirement: Stage root configs are typed and method-specific
The system SHALL model script and workflow stage configuration as typed stage root configs, and SHALL model retrieval and training jobs as method-specific discriminated settings unions instead of wide optional method-family bags.

#### Scenario: Retrieval job settings are method-specific
- **WHEN** a BM25 retrieval config is structured
- **THEN** its settings do not contain dense encoder fields
- **WHEN** a checkpoint graph retrieval config is structured
- **THEN** its settings do not contain graph-rerank settings fields

#### Scenario: Pair build config owns sampling settings
- **WHEN** `scripts/build_train_pairs.py` loads config
- **THEN** negative sampling and hard dense encoder settings are represented inside `PairBuildStageConfig.job`
- **THEN** direct CLI overrides take precedence over file config values

#### Scenario: Evaluate config is method-agnostic
- **WHEN** `scripts/evaluate_retrieval.py` loads config
- **THEN** the evaluate stage config consumes prediction, label, graph, output, and failure-case settings without depending on retrieval method internals

#### Scenario: Train stage uses training registry
- **WHEN** `scripts/train_graph_retriever.py` starts training from a typed stage config
- **THEN** the train stage runner calls `Registry.training.build(config.job, deps)`
- **THEN** adding a future training method does not require method-specific branches in the train stage runner

### Requirement: Workflow planning uses typed projections
The system SHALL make workflow manifest generation prefer typed stage config projections while keeping existing manifest JSON readable and preserving downstream script invocation through argv.

#### Scenario: Manifest stores resolved stage configs
- **WHEN** an experiment manifest is initialized or refreshed by supported workflow commands
- **THEN** the manifest remains readable by existing workflow code
- **THEN** command generation can use resolved typed stage config projections for downstream script argv

#### Scenario: Ablation patches come from registry
- **WHEN** workflow planning applies an R-GCN ablation variant such as `wo_bridge`
- **THEN** the typed patch comes from `graph_memory.registry.ablations`
- **THEN** `scripts/workflow/registry.py` does not independently own R-GCN variant semantics

### Requirement: Config schema cleanup preserves compatibility
The system SHALL support a schema v2 method config shape with shallow method files and no `defaults` wrapper, while retaining compatibility for existing training config paths during migration.

#### Scenario: Schema v2 omits defaults wrapper
- **WHEN** a schema v2 method config is loaded
- **THEN** the root object acts as the base config
- **THEN** `profiles` remains a patch map and `default_profile` remains the fixed default-profile key

#### Scenario: Existing training config path remains usable
- **WHEN** a user or workflow references `configs/training/dense_rgcn_graph_retriever/base.json`
- **THEN** the config loader can still resolve and structure that config through the compatibility adapter
- **THEN** scripts and stage runners do not need to know whether the old or new path was used

### Requirement: Old dispatch and dict slicing helpers are removed from production paths
The system SHALL remove production runtime dependence on old method-string dispatch and old training config dict-slicing helpers after typed configs and registry builders cover the relevant stages.

#### Scenario: Residual dispatch search is bounded
- **WHEN** production code is scanned for public method string dispatch and `builder_id`
- **THEN** remaining occurrences are limited to registry-owned metadata, compatibility projections, docs, or tests

#### Scenario: Training helper calls are absent from production paths
- **WHEN** production code is scanned for old training config slicing helper calls
- **THEN** pair-build and training execution paths use typed settings instead of slicing untyped training config dictionaries

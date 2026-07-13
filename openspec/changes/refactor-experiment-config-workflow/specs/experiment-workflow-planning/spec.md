## ADDED Requirements

### Requirement: Public experiment surface is five entrypoints
The system SHALL expose `experiment/plan.py`, `experiment/run.py`, `experiment/status.py`, `experiment/inspect.py`, and `experiment/reset.py`. Positional experiment commands, a standalone init command, configurable run roots, and implicit destructive flags MUST NOT remain.

#### Scenario: Implicit initialization
- **WHEN** plan or run is invoked for a new valid name
- **THEN** it SHALL initialize the run layout, resolved config, overrides, run state, and stage configs without requiring a separate init command

#### Scenario: Destructive reset
- **WHEN** a user needs to delete an existing named run
- **THEN** deletion SHALL be available only through `experiment/reset.py name=<name>`

### Requirement: RunLayout exclusively owns run-local paths
The system SHALL use one `RunLayout` to derive every run-local path, including state, resolved config, overrides, stage configs, artifacts, ablation variants, and aliases. Other components MUST consume those paths rather than reconstructing them.

#### Scenario: Stage artifact binding
- **WHEN** the planner binds an output artifact to a stage invocation
- **THEN** the path SHALL originate from `RunLayout` and be absolute before subprocess execution

#### Scenario: Custom root attempt
- **WHEN** a user attempts to configure a run root other than `runs/`
- **THEN** configuration validation SHALL reject the unsupported key or value

### Requirement: Method registry is the single dependency authority
The typed method registry SHALL define public method identity, lifecycle, graph source, tuning dependency, training dependency, checkpoint kind, seed method, and artifact roles for all eight supported methods. YAML MUST NOT duplicate runtime dependency edges.

#### Scenario: Seeded Dense-FT R-GCN
- **WHEN** `dense_ft_rgcn_graph_retriever` is selected without `dense_ft` as a public output method
- **THEN** the plan SHALL include Dense-FT pair and train dependencies before the R-GCN training and retrieval stages

#### Scenario: Graph rerank dependency
- **WHEN** a graph-rerank retrieval invocation is planned
- **THEN** it SHALL bind the selected tuning config produced for the correct method and dev split

### Requirement: WorkflowPlanner builds typed ordered invocations
The planner SHALL build a typed stage-invocation DAG and a stable executable ordering from the validated config, method registry, stage bounds, and ablation selection. Each invocation SHALL identify stage, method, split, variant, resolved YAML path, script path, declared inputs, declared outputs, and dependencies.

#### Scenario: Plan display
- **WHEN** a user invokes plan
- **THEN** output SHALL show every low-level script, stage, method, split, variant, and complete argv in execution order without executing a stage

#### Scenario: Run parity
- **WHEN** run executes an invocation
- **THEN** it SHALL print the same command representation produced by plan immediately before subprocess launch

### Requirement: Stage range selection validates dependencies
The planner SHALL support `stages.from` and `stages.to` over legal workflow order. Starting after a prerequisite stage SHALL be allowed only when the required bound artifact is already valid; otherwise planning SHALL fail with the missing path and dependency role.

#### Scenario: Resume from retrieval with checkpoint
- **WHEN** retrieval is selected as the first stage and the required checkpoint and summary are valid
- **THEN** the planner SHALL allow the range without re-adding training

#### Scenario: Missing checkpoint
- **WHEN** retrieval is selected as the first stage but the required checkpoint is absent or stale
- **THEN** planning SHALL fail before launching subprocesses and identify the checkpoint path

### Requirement: Dataset and method plan parity is preserved
The planner SHALL preserve the approved stage dependencies and output roles for HotpotQA, 2WikiMultiHopQA, and MuSiQue, all eight public methods, the dedicated Memory Stream path, and the approved default seven-method quick plan.

#### Scenario: Default quick plan
- **WHEN** the default root configuration is planned
- **THEN** the plan SHALL match the frozen HotpotQA quick method set and stage/dependency snapshot

#### Scenario: Dataset adapters
- **WHEN** the same supported method is planned for each supported dataset
- **THEN** the invocation SHALL bind the selected dataset's projector, split source, and capacity-derived count without changing method semantics

### Requirement: Domain ablation semantics are preserved
The planner SHALL keep registered changed-dimension invalidation, per-variant artifact namespaces, baseline aliases, ordinary-baseline inclusion, and aggregate inclusion. `ablation.enable` SHALL be the sole on/off gate. The default `ablation.variants` list SHALL explicitly enumerate every executable variant, and every configured variants list SHALL be non-empty and validated. Hydra multirun MUST NOT replace domain ablations.

#### Scenario: All variants
- **WHEN** `ablation.enable=true` is selected without overriding the default variants list
- **THEN** every executable registered variant SHALL be planned and `full_rgcn` SHALL be included only as a baseline alias

#### Scenario: Selected variants
- **WHEN** `ablation.enable=true` is selected with a non-empty list of valid variant identifiers
- **THEN** only those executable variants SHALL be planned and any unknown identifier SHALL fail config validation

#### Scenario: Empty variants
- **WHEN** an author configures an empty `ablation.variants` list
- **THEN** config validation SHALL fail instead of assigning an implicit disabled or all-variants meaning

### Requirement: Named run identity rejects incompatible reuse
The system SHALL persist normalized resolved configuration and run mode in typed state. Reusing a name with a different config or switching between single and multirun modes SHALL fail without deleting or mutating prior artifacts.

#### Scenario: Exact resume
- **WHEN** an existing name is opened with the same normalized config and mode
- **THEN** plan or run SHALL reuse its state and evaluate live artifact status

#### Scenario: Config mismatch
- **WHEN** an existing name is opened with any different validated config
- **THEN** the entrypoint SHALL fail and direct the user to choose a new name or explicitly reset the old run


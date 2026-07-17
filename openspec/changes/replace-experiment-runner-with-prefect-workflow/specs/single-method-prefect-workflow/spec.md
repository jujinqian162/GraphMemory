## ADDED Requirements

### Requirement: One Hydra job selects one final method
The experiment configuration SHALL select exactly one final retrieval method and SHALL reject a list of final methods. A method that supports ablation SHALL select exactly one effective variant, while a method without variants MUST reject variant configuration.

#### Scenario: Ordinary final method
- **WHEN** a user composes an experiment with `method=bm25`
- **THEN** the resolved job contains exactly the BM25 final method and no ablation variant

#### Scenario: R-GCN variant
- **WHEN** a user composes an R-GCN experiment with `method.variant=wo_graph`
- **THEN** the resolved job contains exactly that final method and the `wo_graph` effective variant

#### Scenario: Retired multi-method list
- **WHEN** a user supplies the retired `methods=[bm25,dense]` contract
- **THEN** configuration validation fails instead of executing more than one final method in the job

#### Scenario: Retired ablation variant list
- **WHEN** a user supplies `ablation.variants=[full_rgcn,wo_graph]` or another list-valued variant contract
- **THEN** configuration validation fails and directs the user to launch one singular-variant job per command or Hydra job

### Requirement: One readable Prefect Flow owns the workflow
The repository SHALL define one Prefect Flow whose ordinary Python control flow directly shows the stage chain for every supported final method. It MUST NOT compile or interpret a separate workflow plan, node registry, generated stage contract, or dependency graph.

#### Scenario: Inspect a method chain
- **WHEN** a developer reads the final-method branch in the Flow
- **THEN** the prepare, optional graph/pair/train dependencies, retrieval, evaluation, tracking, and output calls are visible in execution order

#### Scenario: Unsupported method
- **WHEN** a method outside the validated final-method union reaches the Flow boundary
- **THEN** the Flow fails explicitly and does not attempt dynamic registry-based planning

### Requirement: Tasks execute in process without stage subprocesses
The Flow SHALL call Prefect Tasks directly and synchronously in the first implementation. Experiment stages MUST NOT be executed through `subprocess`, generated stage YAML, script `main()` functions, Prefect deployments, or distributed Task runners.

#### Scenario: Execute a smoke workflow
- **WHEN** a user runs `experiment/run.py` for a smoke experiment
- **THEN** every selected scientific stage executes through directly called Prefect Tasks in the Flow process

#### Scenario: Model service reuse
- **WHEN** two directly called stages require the same frozen encoder identity in one Flow process
- **THEN** the existing process-local encoder service may reuse the loaded model without a subprocess boundary

### Requirement: Stage services remain independent of orchestration
Scientific stage implementations SHALL be importable functions under `graph_memory/stages/` that accept typed scientific inputs or processed artifact references. Prefect decorators, cache settings, MLflow run IDs, run output paths, and CLI parsing MUST NOT become required domain-stage inputs.

#### Scenario: Direct stage test
- **WHEN** a stage service is tested without Prefect
- **THEN** it can execute from typed domain inputs or processed references without constructing a Flow, CLI invocation, or run directory

### Requirement: Composite dependencies are explicit
A final method that requires another trainable method SHALL call that dependency explicitly in its Flow branch. The workflow MUST NOT infer hidden train dependencies through a generic traversal algorithm.

#### Scenario: Dense-FT-seeded R-GCN
- **WHEN** the selected final method is Dense-FT-seeded R-GCN
- **THEN** the Flow explicitly builds Dense-FT training data, trains or reuses Dense-FT, and passes its processed model reference into R-GCN training before final retrieval

#### Scenario: Dense-FT is not a separate final result
- **WHEN** Dense-FT runs only as the seed dependency of R-GCN
- **THEN** the job still produces one final Dense-FT-seeded R-GCN result and does not create a second final-method run

### Requirement: Multi-baseline and ablation studies use independent jobs
The system SHALL express a set of baselines or variants as independently launched Hydra jobs. Jobs MAY be launched by Hydra multirun, separate user commands, or an external launcher. Every job SHALL select one singular method/variant, run in its own process, create an independent Prefect Flow run and final MLflow run, and remain groupable under the same user-visible study name. The experiment Flow MUST NOT expand an ablation variant list or use `task.submit()` to fan variants out.

#### Scenario: Baseline sweep
- **WHEN** a user launches a Hydra multirun over BM25, Dense, and GraphRAG
- **THEN** Hydra creates three independent single-method jobs rather than one Flow with three final branches

#### Scenario: Ablation sweep
- **WHEN** a user launches a Hydra multirun over `full_rgcn`, `wo_graph`, and `wo_edge_type`
- **THEN** each variant becomes one independent Flow and may reuse compatible prepared, graph, pair, or training inputs from the shared cache

#### Scenario: Manual multi-GPU ablation
- **WHEN** a user launches one singular-variant command on `cuda:0` and another singular-variant command on `cuda:1`
- **THEN** the commands run as independent processes and peer MLflow runs while sharing any compatible upstream Prefect cache results

#### Scenario: Concurrent variants reach a shared cache boundary
- **WHEN** independently launched variants request an identical uncached upstream result at the same time
- **THEN** opaque atomic asset publication prevents overwrite even if first computation is duplicated, while the first variant-specific Task uses a distinct cache identity and output asset

### Requirement: Planner-era execution controls are absent
The public experiment interface MUST NOT expose a plan command, arbitrary stage selection, `stages.from`, `stages.to`, generated stage execution commands, or completed-prefix resume. Re-running the complete Flow SHALL be the recovery and downstream-recompute interface.

#### Scenario: Re-run after a failure
- **WHEN** a user repeats the full experiment command after a Task failure
- **THEN** Prefect reuses compatible completed Tasks and runs the failed or invalidated work without a stage-range argument

#### Scenario: Retired plan command
- **WHEN** active documentation and public entrypoints are inspected after cutover
- **THEN** they do not advertise or provide `experiment/plan.py`

### Requirement: Cross-run aggregation is outside the computation Flow
The single-method Flow SHALL produce one final metric record and MUST NOT execute a workflow aggregate stage or ablation index stage. Cross-run tables SHALL be produced through MLflow comparison or a separate read-only reporting/export boundary.

#### Scenario: Complete one experiment
- **WHEN** one final method finishes evaluation
- **THEN** the Flow completes with that method's result without reading metrics from other experiments

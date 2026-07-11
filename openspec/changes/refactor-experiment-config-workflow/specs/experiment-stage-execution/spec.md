## ADDED Requirements

### Requirement: Stage scripts accept one resolved YAML input
Every low-level prepare, graph, pair, tune, train, retrieve, evaluate, and aggregate script SHALL accept exactly `--config <path>` as its public scientific input and validate that YAML with the stage's discriminated Pydantic model. Scripts MUST NOT provide scientific argparse defaults, legacy JSON loaders, or independent override flags.

#### Scenario: Direct stage execution
- **WHEN** a developer invokes a stage script with a valid resolved YAML path
- **THEN** the script SHALL execute independently and produce the same local artifact and summary contracts used by the experiment runner

#### Scenario: Unknown stage field
- **WHEN** a stage YAML contains an unknown field
- **THEN** the script SHALL fail validation before invoking domain work

### Requirement: Stage configs are method-specific discriminated unions
Stage configuration contracts SHALL make variant-specific dependencies required only for variants that consume them. Stateless BM25/Dense, graph rerank, Memory Stream, R-GCN, Dense-FT, and Dense-FT-seeded R-GCN MUST NOT share a wide optional input bag.

#### Scenario: Graph-rerank retrieval
- **WHEN** a graph-rerank retrieval config is validated
- **THEN** graph input and selected tuning config SHALL be required

#### Scenario: Plain BM25 retrieval
- **WHEN** a BM25 retrieval config is validated
- **THEN** graph, selected-config, checkpoint, encoder, and device fields SHALL not be accepted

### Requirement: Shared lifecycle writes stage summaries consistently
A shared execution helper SHALL own start time, success/failure handling, atomic `StageRunSummary` writes, timing/count observations, optional active child-run mirroring, and exception propagation for every stage script.

#### Scenario: Summary parity across stages
- **WHEN** any supported stage succeeds
- **THEN** its summary SHALL use the same typed status, timestamps, input/output, config, and error field semantics

#### Scenario: Domain exception
- **WHEN** domain execution raises
- **THEN** the helper SHALL preserve the original exception while recording failure state

### Requirement: Scientific and artifact validators remain authoritative
The migration SHALL retain dataset leakage checks, split boundaries, projector validation, graph validation, selected-config validation, checkpoint validation, request-boundary validation, and output schema validation outside the configuration-shape layer.

#### Scenario: Test leakage attempt
- **WHEN** a test retrieval stage is given dev labels or gold-only edges as input-visible data
- **THEN** existing domain/leakage validation SHALL reject the request even if the YAML shape is valid

#### Scenario: Invalid checkpoint shape
- **WHEN** a stage binds a file where its method requires a Dense-FT model directory
- **THEN** artifact/checkpoint validation SHALL reject it before scientific execution

### Requirement: Local artifacts remain the scientific source of truth
Prepared records, graphs, train pairs, selected tuning configs, checkpoints/model directories, predictions, failure cases, stage-local metric JSON/JSONL, and aggregate CSVs SHALL remain local artifacts with stable schemas. MLflow MUST be treated only as a mirror.

#### Scenario: Direct training without tracking parent
- **WHEN** a training stage runs directly without an experiment parent context
- **THEN** it SHALL write training metric JSONL and checkpoint artifacts and SHALL not create an orphan MLflow run

#### Scenario: Aggregate delivery
- **WHEN** aggregate succeeds
- **THEN** `main_results.csv`, `path_results.csv`, `efficiency_results.csv`, and applicable `ablation_results.csv` SHALL be portable without access to MLflow

### Requirement: Process isolation and fail-fast execution are preserved
The experiment runner SHALL execute stage invocations sequentially as subprocesses, print each planned command before launch, stop on the first nonzero exit, and update local run state/status from artifacts after execution.

#### Scenario: Stage failure
- **WHEN** a stage subprocess exits nonzero
- **THEN** no later invocation SHALL start and the failed output SHALL not be considered complete

#### Scenario: GPU stage completion
- **WHEN** a GPU-using stage process exits
- **THEN** subsequent stages SHALL run in a fresh process rather than sharing the prior runtime

### Requirement: Output behavior remains compatible at the scientific contract level
The new executor SHALL preserve prediction, failure-case, metric, checkpoint metadata, selected-config, and aggregate table schemas and SHALL preserve the supported method/dataset workflow semantics except for explicitly changed public CLI/config behavior.

#### Scenario: Deterministic method parity
- **WHEN** a frozen deterministic-method smoke fixture runs through legacy and new implementations
- **THEN** ranked predictions and logical metrics SHALL match exactly

#### Scenario: Trainable method parity
- **WHEN** a trainable-method comparison runs under the declared seed and device policy
- **THEN** plan/config/artifact schemas SHALL match and scientific metrics SHALL remain within predeclared tolerances


## ADDED Requirements

### Requirement: Public experiment surface is five files
The supported public CLI SHALL be `experiment/plan.py`, `experiment/run.py`, `experiment/status.py`, `experiment/inspect.py`, and `experiment/reset.py`. Internal modules under `graph_memory/experiment/` SHALL provide reusable behavior but MUST NOT be the only documented or executable public path.

#### Scenario: Plan command
- **WHEN** a user runs `python experiment/plan.py name=<name>`
- **THEN** the command SHALL compose, validate, initialize if needed, and print the typed plan without executing a stage

#### Scenario: Run command
- **WHEN** a user runs `python experiment/run.py name=<name>`
- **THEN** the command SHALL execute the same plan representation and preserve completed-prefix resume

#### Scenario: Missing public file
- **WHEN** the repository public-command contract is tested
- **THEN** all five file paths SHALL exist and execute; module-only replacements SHALL not satisfy the requirement

### Requirement: Public commands use key-value arguments
The five commands SHALL accept the documented `key=value` spelling. Plan and run SHALL use Hydra composition; status, inspect, and reset SHALL use narrow closed command parsing without positional aliases or legacy argparse experiment subcommands.

#### Scenario: Method override
- **WHEN** a user supplies `methods='[bm25,dense]'` to plan or run
- **THEN** only those public result methods SHALL be selected while hidden dependencies remain registry-derived

#### Scenario: Reset command
- **WHEN** a user supplies `name=<name>` to reset
- **THEN** reset SHALL validate and delete only the contained named run without starting an experiment composition job

### Requirement: Public docs describe one command path
README and active operations documentation SHALL use the five top-level file entrypoints, flattened config paths, fixed MLflow location, and current overrides. They MUST NOT advertise module-only entrypoints, deleted JSON configs, `_base`, root presets, or `mlruns/`.

#### Scenario: Active documentation scan
- **WHEN** active user-facing documentation is scanned
- **THEN** command examples and paths SHALL agree with the public interface and canonical config tree

### Requirement: Public behavior preserves verified workflows
The interface correction SHALL preserve the default seven-method HotpotQA quick workflow, all-eight-method smoke workflow, supported 2Wiki and MuSiQue workflows, Memory Stream input path, hidden Dense-FT dependency, ablations, stage bounds, cache-disabled rerun, multirun, status, reset, and local artifact delivery.

#### Scenario: Default quick parity
- **WHEN** the corrected public run command executes the default quick configuration
- **THEN** method set, stage order, dependencies, typed configs, artifact roles, and scientific outputs SHALL match the frozen accepted behavior

#### Scenario: Direct script parity
- **WHEN** a generated stage YAML is executed directly
- **THEN** it SHALL produce the same local artifacts and summary contract as orchestrated execution without creating an orphan MLflow run


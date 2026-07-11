## ADDED Requirements

### Requirement: Hydra is the sole experiment composer
The system SHALL compose experiment inputs only at public experiment entrypoints using Hydra configuration groups and `key=value` overrides. Low-level stage scripts MUST NOT start Hydra jobs.

#### Scenario: Dataset group replacement
- **WHEN** a user invokes an experiment entrypoint with `dataset=2wiki`
- **THEN** Hydra SHALL replace the complete dataset group, including its adapters, split sources, and capacities

#### Scenario: Low-level script invocation
- **WHEN** the workflow invokes a stage script
- **THEN** the script SHALL receive one resolved YAML path and SHALL NOT perform Hydra composition

### Requirement: Root configuration is complete and default-owned
The root experiment configuration SHALL contain `name`, dataset and profile selections, all method configurations, public methods, seed, device, top-k, stage bounds, cache policy, ablation policy, search spaces, and tracking configuration. Business defaults MUST exist in YAML and MUST NOT be supplied by application code.

#### Scenario: Missing name
- **WHEN** the user invokes plan or run without a `name`
- **THEN** composition or validation SHALL fail before run directories or MLflow runs are created

#### Scenario: Default experiment
- **WHEN** the user supplies only `name=demo`
- **THEN** the resolved configuration SHALL represent HotpotQA, the quick profile, and the approved seven default methods excluding Memory Stream

### Requirement: Pydantic validates resolved primitive inputs
The system SHALL resolve OmegaConf interpolation to primitive containers and validate the result with Pydantic V2 closed models. Unknown keys, unresolved mandatory values, invalid discriminators, invalid cross-field values, and bool/string coercion into scientific numeric fields MUST be rejected before planning.

#### Scenario: Unknown key
- **WHEN** a composed configuration contains a key not declared by its Pydantic model
- **THEN** validation SHALL report the unknown key and SHALL NOT build a stage plan

#### Scenario: Invalid scientific scalar
- **WHEN** a scientific numeric field receives a boolean or numeric string
- **THEN** validation SHALL reject the value instead of coercing it

#### Scenario: Unresolved value
- **WHEN** any resolved configuration value remains `???`
- **THEN** validation SHALL fail before writing a run state

### Requirement: Dataset capacity and profile policy are separate
Dataset groups SHALL define validated logical-window capacities for train, dev, and test, while profile groups SHALL define `fixed` or `all_available` count policies and trainable-method settings. The resolver SHALL combine them and reject out-of-range fixed windows.

#### Scenario: Cloud full profile
- **WHEN** `profile=cloud-full` is selected
- **THEN** train and test SHALL use the selected dataset capacity, dev SHALL use the declared fixed count of 500, and the profile's trainable-method settings SHALL be applied

#### Scenario: Fixed window exceeds capacity
- **WHEN** a profile's offset plus fixed count exceeds a dataset split capacity
- **THEN** validation SHALL fail before any stage executes

### Requirement: Root overrides have one propagation path
The root `seed` and `device` SHALL be the only global sources for their respective concerns. The typed resolver SHALL propagate seed to every random consumer and device only to method variants that require a device, without fallback to method-local defaults.

#### Scenario: Seed override
- **WHEN** the user supplies `seed=14`
- **THEN** prepare, pair construction, tuning, R-GCN training, and Dense-FT training stage configs SHALL all receive seed 14 where they consume randomness

#### Scenario: CPU-only method
- **WHEN** BM25 is planned with a root device override
- **THEN** its discriminated stage config SHALL not gain a meaningless device field

### Requirement: Method configs are composed independently from method selection
Hydra SHALL compose all eight method configuration groups into stable `method_configs.<method>` keys. The public `methods` list SHALL determine execution only, and hidden dependencies SHALL be resolved by the typed method registry rather than by Hydra defaults or `_target_` metadata.

#### Scenario: Method-specific override
- **WHEN** a user overrides `method_configs.dense_ft.train.trainer.learning_rate=3e-5`
- **THEN** only the Dense-FT trainer configuration SHALL change

#### Scenario: Method subset
- **WHEN** a user supplies `methods=[bm25,dense]`
- **THEN** only those public methods and their required shared stages SHALL be planned

### Requirement: Public presets and inspection replace arbitrary JSON selection
The system SHALL support normal experiments through groups and overrides and exceptional combinations through named Hydra root presets. Inspect SHALL discover datasets, profiles, configs, methods, stages, and ablations; it MUST NOT expose a recipe concept.

#### Scenario: Special preset
- **WHEN** the user selects `--config-name=2wiki_tiny`
- **THEN** Hydra SHALL compose the committed root preset and inspect SHALL list it as a config

#### Scenario: Recipe inspection
- **WHEN** a user requests inspect output
- **THEN** no recipe kind or recipe list SHALL be available

### Requirement: Hydra job paths and launch mode are explicit
Hydra SHALL use BasicLauncher sequentially with `hydra.job.chdir=false`. Single runs SHALL use `runs/${name}`; multirun jobs SHALL use Hydra job number plus override dirname below the named multirun root.

#### Scenario: Working directory stability
- **WHEN** plan or run starts a Hydra job
- **THEN** repository-relative inputs SHALL resolve from the original working directory and subprocess configs SHALL contain normalized absolute paths

#### Scenario: Sequential multirun
- **WHEN** a user invokes Hydra multirun
- **THEN** jobs SHALL execute sequentially and each job SHALL receive a distinct path derived from job number and override dirname


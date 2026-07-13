## ADDED Requirements

### Requirement: Configuration has one canonical authoring root
The system SHALL compose experiments from `configs/config.yaml` and meaningful config groups directly below `configs/`. The system MUST NOT require an intermediate `configs/experiment/` namespace, `_base.yaml`, or a forwarding root config.

#### Scenario: Default composition
- **WHEN** a user runs the default experiment with only `name=<name>`
- **THEN** Hydra SHALL compose the complete validated configuration from `configs/config.yaml` without another root layer

#### Scenario: Author inspects the config tree
- **WHEN** an author lists files below `configs/`
- **THEN** the canonical root and each real user-choice group SHALL be directly discoverable without entering an experiment-only namespace

### Requirement: Config groups represent real choices only
The system SHALL retain groups only for dimensions with multiple supported user-selectable configurations. A group with one fixed option MUST be folded into the canonical root or its owning config, and runtime method dependency metadata MUST NOT be copied into YAML.

#### Scenario: Fixed tracking policy
- **WHEN** the repository supports only the fixed SQLite tracking policy
- **THEN** tracking values SHALL be owned by the root config rather than a one-option `tracking/default.yaml` group

#### Scenario: Graph-rerank seed method
- **WHEN** a graph-rerank method is composed
- **THEN** its seed-method dependency SHALL come from the runtime method registry and SHALL NOT appear as duplicate method-config YAML

### Requirement: Ordinary overrides do not create root presets
Dataset, profile, method, stage, cache, ablation, seed, device, top-k, and method-specific scientific changes SHALL be expressed through config-group selection or CLI overrides. The system MUST NOT retain a root preset that only saves such overrides.

#### Scenario: 2Wiki tiny run
- **WHEN** a user requests a small 2Wiki workflow
- **THEN** the workflow SHALL be expressible with `dataset=2wiki`, a retained size profile, method selection, and any explicit split override without a `2wiki_tiny.yaml` root

#### Scenario: R-GCN ablation run
- **WHEN** a user requests all registered R-GCN ablations
- **THEN** the workflow SHALL be expressible with method and `ablation.enable=true` overrides without enumerating variants or using a dataset-specific ablation root config

### Requirement: Special input shapes belong to the dataset contract
An importance-backed HotpotQA split, if retained, SHALL be represented as a dataset config option with complete source, label, capacity, and capability fields. It MUST NOT be a root preset that also selects methods and a profile.

#### Scenario: Memory Stream dataset selection
- **WHEN** Memory Stream uses canonical inputs, canonical labels, and an importance artifact
- **THEN** dataset selection SHALL provide the complete input binding while method and profile selection remain independent

### Requirement: Method configs use readable multi-select composition
The canonical defaults list SHALL compose all required method configs through Hydra multi-select. It MUST NOT enumerate per-method package override entries such as `method_configs@method_configs.<name>`.

#### Scenario: Root config review
- **WHEN** an author reads the canonical defaults list
- **THEN** all composed method configs SHALL be visible as one multi-select group and the resolved config SHALL expose each method under `method_configs.<method>`

### Requirement: Narrow commands require no command YAML
Status, inspect, and reset SHALL validate their `key=value` inputs with closed command models without creating Hydra jobs or loading `status_command.yaml`, `inspect_command.yaml`, or `reset_command.yaml`.

#### Scenario: Status inspection
- **WHEN** a user runs `python experiment/status.py name=<name>`
- **THEN** status SHALL read the named run without creating a Hydra output directory or requiring a command config file

#### Scenario: Missing command field
- **WHEN** a required status, inspect, or reset field is omitted
- **THEN** the command boundary SHALL fail clearly without persisting `???` as a runtime value

### Requirement: Configuration contains no dead fields
Every persisted config field SHALL have a production consumer. Fields used only by composition tests, copied without consumption, or retained from removed behavior MUST be deleted.

#### Scenario: Residual config scan
- **WHEN** the simplified config models and YAML are scanned
- **THEN** unused dataset adapter, duplicate seed-method, unused training-curve rendering, and unused aggregate input-directory fields SHALL be absent


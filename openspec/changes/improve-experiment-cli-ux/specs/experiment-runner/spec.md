## ADDED Requirements

### Requirement: Method-first workflow planning
The experiment runner SHALL derive the default stage plan from the selected retrieval method or methods without requiring users to enumerate necessary stages.

#### Scenario: Flat method uses complete flat workflow
- **WHEN** a user plans or runs an experiment with method `bm25` and no stage selection
- **THEN** the selected workflow includes `prepare`, `graphs`, `retrieve`, `evaluate`, and `aggregate`
- **THEN** the workflow does not include `pairs`, `tune`, or `train`

#### Scenario: Graph-rerank method includes tuning
- **WHEN** a user plans or runs an experiment with method `dense_graph_rerank` and no stage selection
- **THEN** the selected workflow includes `prepare`, `graphs`, `tune`, `retrieve`, `evaluate`, and `aggregate`

#### Scenario: Trainable graph method includes supervision and training
- **WHEN** a user plans or runs an experiment with method `dense_rgcn_graph_retriever` and no stage selection
- **THEN** the selected workflow includes `prepare`, `graphs`, `pairs`, `train`, `retrieve`, `evaluate`, and `aggregate`
- **THEN** the selected workflow does not require `dense_graph_rerank` tuned config artifacts unless the method registry declares such a dependency

### Requirement: Stage range selection
The experiment runner SHALL support contiguous stage ranges using a start stage and optional end stage over the selected method workflow.

#### Scenario: Select a flat-method range
- **WHEN** a user plans method `bm25` from `prepare` to `retrieve`
- **THEN** the selected stages are `prepare`, `graphs`, and `retrieve`

#### Scenario: Select a trainable-method range
- **WHEN** a user plans method `dense_rgcn_graph_retriever` from `prepare` to `retrieve`
- **THEN** the selected stages are `prepare`, `graphs`, `pairs`, `train`, and `retrieve`

#### Scenario: Reject range outside selected workflow
- **WHEN** a user plans method `bm25` from `tune`
- **THEN** the runner fails before command execution
- **THEN** the error lists the stages available for that selected workflow

### Requirement: Contract-name config resolution
The experiment runner SHALL accept public config names at the top-level interface and resolve them through documented config directories.

#### Scenario: Experiment config name resolves to config file
- **WHEN** a user passes experiment config name `hotpotqa_evidence_retrieval`
- **THEN** the runner loads `configs/experiments/hotpotqa_evidence_retrieval.json`

#### Scenario: Training config name resolves under method directory
- **WHEN** an experiment config references training config `base` for method `dense_rgcn_graph_retriever`
- **THEN** the runner loads `configs/training/dense_rgcn_graph_retriever/base.json`

#### Scenario: Explicit config paths remain supported
- **WHEN** a user passes an existing JSON file path as an experiment config
- **THEN** the runner loads that path without rewriting it as a contract name

### Requirement: Discoverable runner resources
The experiment runner SHALL provide read-only CLI subcommands that list resources users can pass back into experiment commands.

#### Scenario: List stages
- **WHEN** a user runs `scripts/experiment.py stages list`
- **THEN** the output lists public stage names in execution order

#### Scenario: List methods
- **WHEN** a user runs `scripts/experiment.py methods list`
- **THEN** the output lists supported retrieval method names from the method registry
- **THEN** each method entry includes its default workflow stages

#### Scenario: List configs
- **WHEN** a user runs `scripts/experiment.py configs list`
- **THEN** the output lists experiment configs, search-space configs, and training configs using public names and file paths

#### Scenario: List profiles for a config
- **WHEN** a user runs `scripts/experiment.py profile list --config hotpotqa_evidence_retrieval`
- **THEN** the output lists profiles from that experiment config with their train, dev, and test example counts
- **THEN** the output lists resolved train, dev, and test split details including source split, max examples, seed, and offset

#### Scenario: Plural profile list alias remains supported
- **WHEN** a user runs `scripts/experiment.py profiles list --config hotpotqa_evidence_retrieval`
- **THEN** the output matches `scripts/experiment.py profile list --config hotpotqa_evidence_retrieval`

### Requirement: Readable plan output
The experiment runner SHALL render plan output as human-readable command blocks rather than a single dense line stream.

#### Scenario: Command blocks include script and parameters
- **WHEN** a user runs the plan command
- **THEN** each generated command block includes the stage, optional method or split qualifier, called script filename, and command arguments
- **THEN** each CLI option appears on its own line
- **THEN** there is a blank line between adjacent command blocks

#### Scenario: Option coloring is presentation-only
- **WHEN** color output is enabled for the plan command
- **THEN** option names beginning with `--` are colorized
- **THEN** command argv values used for execution are unchanged

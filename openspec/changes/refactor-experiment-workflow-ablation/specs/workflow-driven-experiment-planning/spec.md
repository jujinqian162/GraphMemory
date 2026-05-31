## ADDED Requirements

### Requirement: Workflow-driven method planning
The experiment runner SHALL derive each retrieval method's ordered lifecycle from a registered workflow adapter instead of inferring the lifecycle from broad runtime flags or planner-level method-name branches.

#### Scenario: Stateless method uses stateless workflow
- **WHEN** a user plans method `bm25`
- **THEN** the planner SHALL obtain the ordered lifecycle from the registered stateless workflow adapter
- **THEN** the planned method-specific lifecycle SHALL include `retrieve` and `evaluate`
- **THEN** the planned lifecycle SHALL NOT include `pairs`, `tune`, or `train`

#### Scenario: Graph rerank method uses graph-rerank workflow
- **WHEN** a user plans method `dense_graph_rerank`
- **THEN** the planner SHALL obtain the ordered lifecycle from the registered graph-rerank workflow adapter
- **THEN** the planned lifecycle SHALL include `tune`, `retrieve`, and `evaluate`

#### Scenario: Trainable graph method uses R-GCN workflow
- **WHEN** a user plans method `dense_rgcn_graph_retriever`
- **THEN** the planner SHALL obtain the ordered lifecycle from the registered R-GCN workflow adapter
- **THEN** the planned lifecycle SHALL include `pairs`, `train`, `retrieve`, and `evaluate`

### Requirement: Typed closed orchestration values
The workflow-planning layer SHALL represent closed orchestration vocabularies with explicit typed values and SHALL validate serialized CLI and config strings at system boundaries.

#### Scenario: Unknown stage fails with valid choices
- **WHEN** a user requests a stage name that is not a registered `StageId`
- **THEN** the runner SHALL fail before executing low-level commands
- **THEN** the error SHALL list the allowed stage values

#### Scenario: Unknown workflow control value fails fast
- **WHEN** config or manifest input contains an unknown closed workflow control value
- **THEN** the runner SHALL reject the input before command execution
- **THEN** the error SHALL identify the invalid value and its allowed values

### Requirement: Declarative workflow-step dependencies
Each workflow step SHALL declare the semantic artifact roles it consumes, the artifact roles it produces, and the change dimensions that invalidate its outputs.

#### Scenario: Pair-sampling change starts at pairs
- **WHEN** a run unit declares `PAIR_SAMPLING` as a changed dimension
- **THEN** the planner SHALL identify `pairs` as the earliest invalidated R-GCN workflow step
- **THEN** upstream `prepare` and `graphs` artifacts SHALL remain reusable

#### Scenario: Model-structure change starts at train
- **WHEN** a run unit declares `MODEL_STRUCTURE` as a changed dimension
- **THEN** the planner SHALL identify `train` as the earliest invalidated R-GCN workflow step
- **THEN** upstream `pairs` artifacts SHALL remain reusable

#### Scenario: Model graph-view change starts at train
- **WHEN** a run unit declares `MODEL_GRAPH_VIEW` as a changed dimension
- **THEN** the planner SHALL identify `train` as the earliest invalidated R-GCN workflow step
- **THEN** graph artifacts SHALL remain reusable without rebuilding graph JSON files

### Requirement: Deterministic artifact reuse aliases
The planner SHALL represent reusable upstream artifacts as explicit aliases to existing artifact references and SHALL allocate variant-local paths only from the earliest invalidated step onward.

#### Scenario: Model-only variant reuses train pairs
- **WHEN** the planner expands R-GCN variant `wo_graph`
- **THEN** the variant SHALL alias the main R-GCN `train.pairs.json`
- **THEN** the variant SHALL allocate its own effective training config, checkpoint, prediction, metric, and debug output paths

#### Scenario: Pair-sampling variant owns train pairs
- **WHEN** the planner expands R-GCN variant `wo_hard_negatives`
- **THEN** the variant SHALL allocate its own `train.pairs.json` and pair-summary paths
- **THEN** the variant SHALL allocate its own downstream training, prediction, metric, and debug output paths

### Requirement: Explicit downstream resume validation
The planner SHALL combine the requested stage range with resolved artifact aliases and current artifact evidence, and SHALL fail fast when a required upstream artifact is neither scheduled nor already valid.

#### Scenario: Resume variants from retrieve after training
- **WHEN** a user runs an ablation-enabled experiment from `retrieve`
- **WHEN** every selected variant has a valid checkpoint artifact
- **THEN** the planner SHALL schedule selected-variant `retrieve`, `evaluate`, and requested downstream commands
- **THEN** the planner SHALL NOT schedule `pairs` or `train`

#### Scenario: Resume variant from retrieve without checkpoint
- **WHEN** a user runs an ablation-enabled experiment from `retrieve`
- **WHEN** any selected non-baseline variant checkpoint is absent
- **THEN** the runner SHALL fail before command execution
- **THEN** the error SHALL identify the missing variant checkpoint path
- **THEN** the planner SHALL NOT silently insert `train`

### Requirement: Preserve explicit low-level command plans
The planner SHALL emit concrete low-level script commands with explicit paths for every scheduled workflow step.

#### Scenario: Variant plan exposes actual lifecycle steps
- **WHEN** a user plans an R-GCN ablation variant from `train` through `evaluate`
- **THEN** plan output SHALL display separate `train`, `retrieve`, and `evaluate` command blocks qualified by method and variant
- **THEN** plan output SHALL NOT replace those commands with an opaque `ablate` command

#### Scenario: Existing low-level scripts remain directly usable
- **WHEN** a user invokes an existing low-level script directly with its documented explicit IO arguments
- **THEN** the script SHALL remain usable without a workflow planner or manifest

### Requirement: Workflow registration consistency
The workflow registry SHALL expose exactly one workflow adapter for each public retrieval method and SHALL reject inconsistent registrations before planning commands.

#### Scenario: Registered methods resolve one workflow
- **WHEN** registry validation runs
- **THEN** every public method from the runtime retrieval registry SHALL resolve to exactly one experiment workflow adapter

#### Scenario: Suite references registered method
- **WHEN** registry validation runs
- **THEN** every ablation suite SHALL reference a method with an experiment workflow registration

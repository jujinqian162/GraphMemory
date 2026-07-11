## ADDED Requirements

### Requirement: Production signatures expose only production modes
An optional parameter, default value, union alternative, or fallback branch SHALL exist only when distinct production callers or a genuine domain state exercise the alternatives. Tests MUST NOT be the sole reason for expanding a production signature.

#### Scenario: Hydra initialization
- **WHEN** plan or run initializes from Hydra
- **THEN** repository-root ownership SHALL have one production path and no unused optional root parameter

#### Scenario: Completed-prefix pruning
- **WHEN** resume invokes prefix pruning
- **THEN** cache enablement SHALL be passed explicitly and the pruning function SHALL NOT supply a test-convenience default

#### Scenario: Stage lifecycle
- **WHEN** a direct stage writes its lifecycle summary
- **THEN** the lifecycle API SHALL NOT expose unused summary-path or child-run-id overrides or a test-only publication hook

### Requirement: Validation has one boundary
Hydra/OmegaConf SHALL resolve the known root `DictConfig` once, and Pydantic SHALL validate the resulting object once. Code MUST NOT retain an unused Mapping input path, repeat root-shape checks already guaranteed by the API, or recursively reimplement missing-value and absolute-path validation after the typed boundary.

#### Scenario: Missing Hydra value
- **WHEN** the composed experiment contains a mandatory unresolved value
- **THEN** OmegaConf/Pydantic SHALL reject it without a second recursive `???` scan

#### Scenario: Direct stage path
- **WHEN** a direct stage loads its persisted execution contract
- **THEN** its typed path fields SHALL enforce the required path contract without a generic recursive object walker

### Requirement: Method and ablation metadata have one owner
The existing runtime method registry SHALL be the sole owner of method lifecycle, graph source, tuning kind, training dependencies, checkpoint kind, and seed method. The ablation registry SHALL be the sole owner of variant patches and invalidation metadata.

#### Scenario: Hidden Dense-FT dependency
- **WHEN** Dense-FT-seeded R-GCN is selected
- **THEN** planning SHALL derive its Dense-FT dependency directly from the runtime registry without an experiment-registry copy

#### Scenario: Ablation patch
- **WHEN** a registered ablation variant is planned
- **THEN** planning SHALL apply the registered typed patch rather than switch on the variant name to recreate it

### Requirement: Stage execution uses one persisted contract
Plan display, subprocess execution, direct script loading, lifecycle summaries, status, and resume SHALL consume one complete typed stage execution contract. Direct scripts MUST NOT reconstruct a second invocation or artifact-binding map.

#### Scenario: Direct stage execution
- **WHEN** a low-level script receives `--config <stage-yaml>`
- **THEN** the YAML SHALL already contain validated identity, payload, summary path, inputs, outputs, and dependency identifiers required by the shared lifecycle

#### Scenario: Binding parity
- **WHEN** a planned stage contract is written and loaded by its direct script
- **THEN** the loaded bindings SHALL equal the planned bindings without an `isinstance` reconstruction registry

### Requirement: Different required shapes use discriminated contracts
Where absence changes which fields are required, the system SHALL model complete discriminated variants instead of a wide model containing optional fields.

#### Scenario: Importance preparation
- **WHEN** a preparation contract has kind `importance`
- **THEN** canonical inputs, canonical labels, importance artifact, count, offset, and outputs SHALL be required and raw-only fields SHALL be absent

#### Scenario: Seeded R-GCN training
- **WHEN** the method is Dense-FT-seeded R-GCN
- **THEN** the Dense-FT model directory SHALL be required rather than represented by an optional seed checkpoint shared with ordinary R-GCN

#### Scenario: Ablation aggregation
- **WHEN** aggregate includes ablation results
- **THEN** ablation index, output, and typed selections SHALL be required together; ordinary aggregate SHALL carry none of them

#### Scenario: Artifact alias
- **WHEN** an artifact is an alias
- **THEN** its source path SHALL be required by an alias-specific contract rather than an optional field on every artifact

### Requirement: Persisted state avoids broad Any
Experiment config, stage execution contracts, run state, stage summaries, and observations SHALL use closed Pydantic/JSON-value contracts and MUST NOT persist broad `dict[str, Any]` fields.

#### Scenario: Stage summary readback
- **WHEN** a stage summary is read from disk
- **THEN** its effective config and observations SHALL validate against closed persisted-value contracts

### Requirement: Migrated scripts do not recreate legacy config carriers
Stage scripts SHALL consume Pydantic execution payloads or explicit required domain arguments directly. They MUST NOT copy those values into legacy `Prepare*Args` objects or restore optionality eliminated by the stage contract.

#### Scenario: Raw preparation
- **WHEN** a raw prepare stage executes with required count and combined output
- **THEN** its implementation SHALL have no unreachable `max_examples is None` or `output_combined is not None` branch

#### Scenario: Retrieval method fields
- **WHEN** a validated retrieval variant executes
- **THEN** the script SHALL access its required graphs, selected config, importance, checkpoint, or model fields directly and SHALL NOT discover them through `getattr`

### Requirement: State and status perform no redundant work
Status SHALL own summary comparison and prerequisite state derivation. Planning MUST reuse that authority. Run state SHALL be written only when persisted state changes meaningfully, and corrupt summaries MUST NOT trigger silent attempt-number fallback.

#### Scenario: Prerequisite check
- **WHEN** planning validates an external prerequisite
- **THEN** it SHALL use the same typed status comparison used by status output rather than a private duplicate implementation

#### Scenario: Multi-stage execution
- **WHEN** several stages execute successfully
- **THEN** run state SHALL NOT be rewritten after every stage solely to change `updated_at`

#### Scenario: Corrupt prior summary
- **WHEN** a stage summary exists but cannot be validated
- **THEN** the stage SHALL be reported stale or corrupt and a new attempt SHALL NOT silently restart at attempt 1

### Requirement: Internal values remain typed across stages
Ablation selections and other repository-generated structured values SHALL remain typed records between planning, execution, persistence, and aggregation. String encoding and parsing SHALL occur only at a user-facing CLI or display boundary.

#### Scenario: Ablation aggregate
- **WHEN** planned ablation selections reach the aggregate stage
- **THEN** method and variant SHALL remain typed fields without repeated `method=variant` partition and validation branches

### Requirement: Dead code and forwarding abstractions are absent
The simplified implementation MUST remove functions, fields, wrappers, defaults, and branches with no production consumer or distinct invariant.

#### Scenario: Residual implementation scan
- **WHEN** the experiment core and migrated scripts are scanned after migration
- **THEN** unused `_looks_like_metric_file`, one-line override/summary wrappers, impossible registry-order checks, no-op exception handlers, unused type alternatives, and the audit-listed optional/default parameters SHALL be absent


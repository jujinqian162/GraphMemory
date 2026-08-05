## ADDED Requirements

### Requirement: Scientific artifact reads are runtime-typed
Every stage that reads a prepared dataset, evidence/provenance graph, training-pair artifact, prediction artifact, checkpoint metadata record, metric artifact, or other scientific payload SHALL validate the decoded value with the owning Pydantic model or `TypeAdapter` before projecting, joining, training, retrieving, evaluating, or reporting it.

#### Scenario: JSON is read for training
- **WHEN** a training stage decodes task, label, graph, pair, or summary JSON
- **THEN** it obtains typed model instances through Pydantic before model loading, tensor materialization, or epoch execution

#### Scenario: Prediction JSON is read for evaluation
- **WHEN** an evaluation stage decodes `predictions.json`
- **THEN** it validates the complete prediction list and native trace union before computing metrics

#### Scenario: Malformed cached payload is referenced
- **WHEN** a declared cached artifact exists but its payload violates the active contract
- **THEN** the consuming stage fails at payload parsing rather than casting it to the expected type

### Requirement: Scientific artifact writes dump validated models
Artifact producers SHALL create validated models before publication and SHALL serialize them using Pydantic JSON-mode dumps or typed adapter dumps. Producers MUST NOT construct an unvalidated dictionary and rely on a later stage-level validator.

#### Scenario: Prepared split is published
- **WHEN** dataset conversion finishes
- **THEN** ranking records, labels, and combined inspection records have already passed their dataset-owned models before JSON is written

#### Scenario: Predictions are published
- **WHEN** retrieval finishes
- **THEN** every prediction and the retrieval batch have already passed their Pydantic contracts before `predictions.json` is written

### Requirement: Valid active artifact schemas remain stable
For every active valid artifact, the migration SHALL preserve field names, required and optional fields, enum/string values, list ordering, numeric values, metric column aliases, trace discriminators, active schema versions, and deterministic JSON content. The change MUST NOT add a legacy translation path for already retired schemas.

#### Scenario: Valid fixture round trips
- **WHEN** a current valid fixture is parsed and dumped through its new model
- **THEN** its normalized JSON value equals the pre-change value

#### Scenario: Current artifact digest is recomputed
- **WHEN** deterministic current-schema data is regenerated from unchanged scientific inputs
- **THEN** sorted JSON publication yields unchanged scientific content and does not change solely because validation moved to Pydantic

#### Scenario: Retired schema is supplied
- **WHEN** an artifact uses a schema version already rejected before this change
- **THEN** the new model rejects it without compatibility conversion or fallback

### Requirement: Dataset split contracts own record and join validation
Each dataset SHALL provide Pydantic ranking, label, and prepared-split aggregate models that preserve candidate ID/position conventions, task prefixes, metadata requirements, gold evidence membership, dependency-edge legality, ranking/label task alignment, and ranking-side label leakage protection.

#### Scenario: Ranking contains a nested gold field
- **WHEN** a ranking record contains a forbidden gold/answer/label key at any protected nested location
- **THEN** the dataset's label-free ranking model rejects it

#### Scenario: Gold evidence is absent from candidates
- **WHEN** a label references an item not declared by its matching ranking record
- **THEN** the prepared-split aggregate rejects the join

#### Scenario: 2Wiki-provenance record is parsed
- **WHEN** a v3 provenance ranking/label pair is parsed
- **THEN** its models preserve schema/construction identity, fixed branch degree, rank-bucket matching, feed confidence fields and mass, binding consistency, gold path, and ranking-label alignment invariants currently enforced by the provenance validator

### Requirement: Graph artifact contracts own topology validation
Evidence and execution-provenance graph models SHALL validate discriminated node shapes, unique node IDs, exactly one query node where required, edge endpoint existence, supported edge types, finite non-negative weights, directed flags, legal provenance type transitions, binding legality, candidate/node coverage, task alignment, and leakage policy.

#### Scenario: Evidence graph misses a request item
- **WHEN** a graph/request aggregate does not contain exactly the request's graph items plus its required query node
- **THEN** Pydantic aggregate validation rejects it

#### Scenario: Provenance transition has illegal node types
- **WHEN** an execution-provenance edge type connects node types outside its declared transition matrix
- **THEN** the provenance graph model rejects it

### Requirement: Training-pair artifacts own supervision validation
Pair record, negative-sampling, summary, and pair-dataset aggregate models SHALL preserve label/sample-type consistency, candidate and optional graph membership, duplicate prevention, exact positive/gold equality, negative/gold exclusion, provenance source precedence, count non-negativity, no-positive-task rejection, and summary/config identity.

#### Scenario: Question node is sampled
- **WHEN** a pair references the query node as a training candidate
- **THEN** the pair aggregate rejects it

#### Scenario: Positive set is incomplete
- **WHEN** materialized positive pairs do not exactly cover a task's gold evidence
- **THEN** the pair aggregate rejects the artifact before training

#### Scenario: Summary disagrees with pair output
- **WHEN** summary counts, sampling configuration, overlap data, or no-positive tasks are inconsistent with the active pair contract
- **THEN** the pair result model rejects publication or consumption

### Requirement: Metric suites own Pydantic row schemas
Each metric suite SHALL expose an authoritative Pydantic aggregate-row model and any per-task/failure-case models it emits. CSV aliases and table column views SHALL derive from model field aliases or declared suite projections rather than an independent global validator column list.

#### Scenario: Evidence metric row is generated
- **WHEN** the evidence suite emits its aggregate row
- **THEN** Pydantic enforces the evidence schema discriminator, required metrics, finite ranges, optional `N/A` metrics, and non-negative latency

#### Scenario: A report selects a column view
- **WHEN** main, path, or efficiency tables select evidence columns
- **THEN** their names derive from the evidence row contract and cannot silently diverge from metric validation

### Requirement: Unchecked scientific casts are prohibited
Production code MUST NOT use `typing.cast` to turn decoded JSON/JSONL/CSV objects into scientific artifact, graph, request, pair, result, checkpoint, or metric types. Casts MAY remain for third-party APIs only when no runtime contract claim is being made.

#### Scenario: Stage source is audited
- **WHEN** stage artifact read paths are inspected by an architecture guard
- **THEN** no read result is cast directly to a scientific contract type without Pydantic validation

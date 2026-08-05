## ADDED Requirements

### Requirement: Scientific value contracts are domain-owned Pydantic models
Every persisted scientific record and every validation-bearing runtime value SHALL have one authoritative Pydantic V2 model in the lowest-level domain that owns the value. The authoritative model SHALL define its fields, defaults, accepted enums or literals, local invariants, and JSON representation without relying on a parallel `TypedDict`, copied field-name set, validation function, or serializer schema.

#### Scenario: Ranked result ownership
- **WHEN** code constructs or reads a ranked result
- **THEN** the retrieval-owned Pydantic result model is both its static type and runtime validator, and no `contracts/ranking.py` declaration is consulted

#### Scenario: Dataset record ownership
- **WHEN** HotpotQA, 2Wiki, MuSiQue, or 2Wiki-provenance ranking and label records are converted or read
- **THEN** the owning dataset's Pydantic record models define the complete accepted shape and local invariants

#### Scenario: Graph and training ownership
- **WHEN** evidence/provenance graphs, train pairs, pair summaries, model configuration, checkpoint metadata, or metric rows are created
- **THEN** the graph, training-pair, model, checkpoint, or evaluation domain owns the authoritative Pydantic model

### Requirement: Scientific models are closed and immutable
Authoritative scientific models SHALL reject unknown fields, validate defaults, and be immutable after construction. Scientific integer, boolean, enum, literal, and finite-number fields MUST reject coercions or non-finite values that would change scientific meaning.

#### Scenario: Unknown field is rejected
- **WHEN** a scientific artifact record contains a field not declared by its owning model
- **THEN** Pydantic validation fails at that model boundary

#### Scenario: Boolean is not accepted as a scientific integer
- **WHEN** a boolean is supplied for an integer count, rank, epoch, seed, or index field
- **THEN** Pydantic validation rejects it rather than using Python's boolean-as-integer relationship

#### Scenario: Non-finite scientific value is rejected
- **WHEN** a score, weight, probability, latency, metric, loss, or checkpoint selection value is NaN or infinite
- **THEN** the owning Pydantic model rejects it

### Requirement: Low-level models are reused instead of copied
Outer experiment configuration, Registry settings and payloads, stage adapters, model trainers, checkpoint writers, and artifact publishers SHALL reuse or compose the low-level authoritative model or its field types. They MUST NOT maintain a second field list, enum allowlist, default set, or structurally equivalent dataclass that can drift independently.

#### Scenario: Retrieval method ID is added
- **WHEN** a new `RetrievalMethodId` becomes valid
- **THEN** ranked-result method validation derives from that enum and requires no independent method allowlist edit

#### Scenario: R-GCN field changes
- **WHEN** a model, training, selection, batching, or inference configuration field changes
- **THEN** the domain model and outer configuration reuse the same declared field/type rather than updating an experiment model, dataclass, manual validator, and JSON conversion independently

#### Scenario: Negative-sampling field changes
- **WHEN** a pair-sampling field or sample type changes
- **THEN** pair construction, summary serialization, and experiment composition consume the authoritative domain model or enum

### Requirement: Custom scientific invariants remain model-owned
Pydantic field and model validators SHALL preserve every currently enforced custom invariant for valid active schemas, including uniqueness, ordering, endpoint membership, exact candidate coverage, label leakage, graph transitions, binding legality, probability and weight mass, connected selection, fallback consistency, positive/negative supervision, metric ranges, and checkpoint identity.

#### Scenario: Existing invalid fixture is replayed
- **WHEN** an invalid fixture currently rejected by `graph_memory.validation` is passed to its replacement model or aggregate contract
- **THEN** it remains rejected for the same scientific reason even if the exception rendering changes to Pydantic's location-aware format

#### Scenario: Existing valid fixture is replayed
- **WHEN** a current valid dataset, graph, pair, prediction, trace, checkpoint metadata, or metric fixture is parsed
- **THEN** it remains valid and produces the same scientific values

### Requirement: Cross-object invariants use explicit aggregate models
An invariant that depends on multiple records or external context SHALL be represented by an explicit Pydantic aggregate/envelope model containing the required typed values. The implementation MUST NOT hide required validation dependencies in an untyped global, copied mapping, or detached `validate_*` service.

#### Scenario: Ranking is checked against its request
- **WHEN** a result is validated for one retrieval request
- **THEN** a typed request/result envelope checks exact candidate coverage, candidate references, retrieved-subgraph references, and task identity

#### Scenario: Labels are checked against rankings
- **WHEN** a prepared split is validated
- **THEN** a dataset-owned aggregate model checks task alignment, gold membership, dependency endpoints, and dataset-specific joins

#### Scenario: Train pairs are checked against supervision
- **WHEN** pair output is validated
- **THEN** a typed pair aggregate contains the requests, labels, optional graphs, pairs, and summary needed to prove supervision correctness

### Requirement: Pydantic is not used as a tensor execution framework
The system SHALL keep runtime tensor shape, dtype, device, offset, batch-isolation, and index-range assertions at explicit computation boundaries rather than forcing them into artifact models. Pure immutable computation records MAY remain dataclasses when they have no separate persisted schema, validation copy, or hand-written serialization contract.

#### Scenario: Invalid tensor offset is detected
- **WHEN** batching or tensorization observes an out-of-range offset or cross-task index
- **THEN** the owning numerical boundary raises its explicit runtime error without requiring a Pydantic tensor model

#### Scenario: Pure computation tuple remains a dataclass
- **WHEN** an internal value is never serialized, externally parsed, or separately validated
- **THEN** this change does not require converting it solely to remove all dataclasses from the repository

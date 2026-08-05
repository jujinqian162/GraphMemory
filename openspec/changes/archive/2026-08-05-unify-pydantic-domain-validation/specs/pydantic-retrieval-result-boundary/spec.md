## ADDED Requirements

### Requirement: Ranked results are closed Pydantic models
The retrieval domain SHALL represent ranked nodes, retrieved subgraphs, result metadata, and complete ranked results with closed Pydantic models. A ranked result SHALL carry a valid Registry method ID, finite scores and latency, non-negative token count and latency, and only declared fields.

#### Scenario: Result is constructed from method output
- **WHEN** a retrieval method returns ranked nodes and trace data
- **THEN** execution constructs a Pydantic ranked-result model rather than a dictionary annotated as a `TypedDict`

#### Scenario: Unsupported method is rejected from the authoritative enum
- **WHEN** a result contains a method value outside `RetrievalMethodId`
- **THEN** Pydantic rejects it without consulting a copied set of method strings

#### Scenario: Result shape is malformed
- **WHEN** a ranked node, retrieved subgraph, metadata object, latency, or token count has an invalid type or unknown field
- **THEN** Pydantic rejects it at result construction or artifact parsing

### Requirement: Per-request ranking invariants are immediate
Retrieval execution SHALL validate each result against the exact request that produced it before appending the result to the completed output collection. The request/result contract SHALL require matching task IDs, every candidate exactly once, no unknown or duplicate candidate IDs, descending score order, and valid retrieved-subgraph node and edge references.

#### Scenario: One method omits a candidate
- **WHEN** a method returns a ranking missing one request candidate
- **THEN** execution fails on that task immediately and does not run later tasks

#### Scenario: One method returns an unsorted ranking
- **WHEN** a task result has a score larger than the preceding score
- **THEN** the per-request Pydantic contract rejects the result before it is appended

#### Scenario: Retrieved edge references an unknown endpoint
- **WHEN** a retrieved-subgraph edge leaves the request candidate/query context
- **THEN** the per-request Pydantic contract rejects it

### Requirement: Native traces use one discriminated Pydantic union
Every trace kind intentionally accepted by the current retrieval boundary SHALL be represented by a closed member of one Pydantic discriminated union keyed by `trace_kind`. This inventory SHALL cover current entity-search/GraphRAG, typed local bridge, execution-provenance, stateless/local execution-provenance, and query-conditioned execution-provenance trace surfaces unless a separate approved method change explicitly retires one.

#### Scenario: Trace is serialized
- **WHEN** a retrieval method emits a native trace model
- **THEN** `model_dump(mode="json")` produces its metadata record without a trace-kind `isinstance` serialization chain

#### Scenario: Trace is parsed
- **WHEN** a prediction artifact contains a native trace
- **THEN** the discriminated union selects its model and validates all declared fields and custom invariants without a hand-written trace-kind dispatch validator

#### Scenario: Trace kind or field is unknown
- **WHEN** a trace has an unsupported discriminator or an undeclared field
- **THEN** Pydantic rejects it

### Requirement: Trace scientific invariants are preserved
Trace member models SHALL retain all currently enforced graph and scientific invariants, including unique records, finite scores, endpoint membership, path adjacency, relation support, normalized affinity and transition mass, PPR coverage and mass, candidate-prize coverage, selection budget, connector legality, selected-path transition identity, selected-edge orientation, selected-subgraph connectivity, accepted/rejected proposal consistency, emitted-edge candidacy, and exact-fallback consistency.

#### Scenario: Probability row is not normalized
- **WHEN** a trace transition row or declared affinity distribution does not sum to one within its owned tolerance
- **THEN** the trace model rejects it

#### Scenario: Selected native subgraph is disconnected
- **WHEN** selected candidates and connectors are not connected by the declared selected native edges
- **THEN** the trace model rejects it

#### Scenario: Fallback contradicts intervention
- **WHEN** exact fallback is true while an accepted proposal, selection step, connector, selected edge, or emitted intervention remains
- **THEN** the owning trace model rejects it

#### Scenario: Binding is attached to the wrong edge type
- **WHEN** a feed edge lacks required binding data or another edge type carries feed-only binding data
- **THEN** the trace or provenance-edge model rejects it

### Requirement: Metadata policy is encoded in the result model
Result metadata SHALL have a declared model surface. Method capability flags owned by the Registry, including the retired `path_metrics_supported` result metadata field, MUST NOT be accepted as arbitrary metadata, while native trace metadata SHALL be typed.

#### Scenario: Registry capability is copied into metadata
- **WHEN** a result contains `metadata.path_metrics_supported`
- **THEN** the result metadata model rejects it and consumers continue to obtain capability from the Registry

### Requirement: Training dev evaluation and production retrieval share result assembly
Evidence R-GCN and provenance R-GCN dev inference SHALL construct ranked results through the same Pydantic result/trace factory and request-relative validation path used by production retrieval. Training code MUST NOT assemble an unchecked result dictionary solely for metric computation.

#### Scenario: Dev inference emits an invalid ranking
- **WHEN** one epoch's dev inference omits, duplicates, misorders, or invents a candidate
- **THEN** the same result contract used in production retrieval fails during that epoch before metrics or checkpoint selection consume it

#### Scenario: Dev and production emit equivalent inputs
- **WHEN** dev and production assembly receive the same request, ranks, subgraph, and trace
- **THEN** they produce equivalent serialized ranked-result records

### Requirement: Batch-level alignment remains explicit and cheap
After all per-task results pass, a typed retrieval-batch contract SHALL verify unique task IDs and exact task coverage against the execution task set. Batch validation MUST NOT defer task-local structural or trace errors that can be detected earlier.

#### Scenario: Duplicate task result is assembled
- **WHEN** two valid per-task results carry the same task ID
- **THEN** final batch validation rejects the duplicate

#### Scenario: Execution produces every task once
- **WHEN** every task has one valid result
- **THEN** the batch contract accepts the collection without reimplementing each result model's local validation

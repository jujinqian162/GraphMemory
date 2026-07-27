## ADDED Requirements

### Requirement: Query-conditioned relation semantics
The non-trained EPGM retriever SHALL compute relation affinity from the request query and frozen schema-owned descriptions for public provenance edge types, SHALL use the configured frozen encoder, and MUST NOT inspect dataset identity, answer labels, or gold evidence.

#### Scenario: Lifecycle query prioritizes revision semantics
- **WHEN** a query is more similar to the frozen descriptions of `invalidates` and `contradicts` than to unrelated relation descriptions
- **THEN** those relation types receive greater transition affinity independent of the dataset name

#### Scenario: Relation descriptions are reproducible
- **WHEN** two runs use the same query, graph, encoder identity, relation-description version, and method config
- **THEN** they produce identical relation similarities and affinities

### Requirement: Recorded confidence remains source-local
The retriever SHALL combine raw recorded edge confidence with type and query affinity before normalizing outgoing arc strength separately for each source node. It MUST NOT globally min-max normalize recorded weights or compare source-local branch probabilities as global path confidence.

#### Scenario: Calibrated feeds siblings retain relative confidence
- **WHEN** two field-bound `feeds` edges leave the same source with weights 0.8 and 0.6 and all other factors are equal
- **THEN** their normalized transition probabilities preserve the 0.8-to-0.6 ordering and ratio before common normalization

#### Scenario: Constant weights are identity
- **WHEN** every outgoing edge under comparison has recorded weight 1.0
- **THEN** recorded confidence does not change their relative transition probabilities

#### Scenario: Zero confidence disables both directions
- **WHEN** a stored edge has recorded weight 0.0
- **THEN** neither a forward nor reverse transition is generated from that edge

#### Scenario: Invalid binding is not traversable
- **WHEN** a `feeds` edge binding does not match its source and target endpoints
- **THEN** the edge receives no transition probability

### Requirement: Typed Personalized PageRank expands over the native graph
The retriever SHALL run deterministic Personalized PageRank over candidate and connector nodes using Dense relevance as teleport mass and query-conditioned typed transitions. It SHALL return convergence diagnostics and MUST bound iteration count.

#### Scenario: Graph-only candidate becomes reachable
- **WHEN** a candidate outside the leading Dense seeds is connected through a positive typed transition to relevant teleport mass
- **THEN** that candidate receives non-zero graph relevance and can enter connected-subgraph selection

#### Scenario: Dangling mass is conserved
- **WHEN** a graph node has no outgoing positive transition
- **THEN** its propagated mass returns through the teleport distribution and every iteration preserves total probability within numerical tolerance

#### Scenario: PPR is deterministic
- **WHEN** the same request and config are evaluated repeatedly
- **THEN** final node masses, iteration count, and residual are identical

### Requirement: Budgeted connected evidence selection
The retriever SHALL jointly select a connected provenance subgraph with no more than `top_k` request-candidate evidence nodes. Non-candidate native graph nodes MAY act as connectors and MUST NOT consume the evidence budget. Already-selected tree arcs SHALL have zero residual cost in later connector searches, and every out-of-budget candidate SHALL pay the opportunity cost of the Dense top-`k` incumbent evidence it displaces.

#### Scenario: Complete dependency chain is selected jointly
- **WHEN** two relevant candidate outputs are connected by a valid `feeds`/`returns` path whose marginal prize exceeds its connector cost
- **THEN** the selected subgraph contains both candidates and the connector path rather than treating the candidates as unrelated promotions

#### Scenario: Connector does not consume evidence budget
- **WHEN** a selected path joins two candidate outputs through a tool-call node that is not a request candidate
- **THEN** the tool-call is retained in the native selected subgraph but only the two output candidates count toward `top_k`

#### Scenario: Evidence budget is enforced
- **WHEN** a candidate path would cause more than `top_k` distinct request candidates to be selected
- **THEN** that expansion is rejected

#### Scenario: Existing connector branch has residual cost only
- **WHEN** a new candidate attaches to a connector already present in the selected tree
- **THEN** its marginal edge cost includes only the newly added branch rather than charging the shared tree path again

#### Scenario: Graph candidate cannot displace stronger Dense evidence for free
- **WHEN** an out-of-budget candidate has positive graph prize but its prize does not exceed new edge cost plus the weakest replaceable Dense incumbent prize
- **THEN** the expansion is rejected and the stronger Dense evidence remains in the top-`k`

#### Scenario: Selection is deterministic under ties
- **WHEN** two expansions have equal marginal objective
- **THEN** stored direction, path cost, original Dense rank, node ids, and edge ids provide a stable total ordering

### Requirement: Exact Dense fallback
The retriever SHALL return the full Dense ranking and scores byte-for-byte when connected extraction cannot produce a positive-marginal multi-candidate selection or violates a numerical, connectivity, or budget invariant. Structural connector paths MAY change ranking without emitting a shared logical dependency edge when their native relations are not contracted dependency types.

#### Scenario: Edgeless candidate graph
- **WHEN** no two request candidates are connected by an admissible provenance path
- **THEN** ranked nodes and scores are exactly Dense and the trace declares exact fallback

### Requirement: Selected edges preserve native provenance direction
The retriever SHALL retain stored edge direction in the native selected subgraph and SHALL collapse only selected candidate-to-candidate paths into public logical edges.

#### Scenario: Reverse traversal does not invert emitted dependency
- **WHEN** selection reaches a candidate by traversing stored edges in reverse
- **THEN** the native edge remains stored-direction oriented and the collapsed candidate dependency is oriented from the path's stored provenance direction

### Requirement: One default architecture across graph realizations
The default non-trained EPGM architecture SHALL be the same query-conditioned diffusion and connected-selection algorithm for synthetic dependency graphs and recorded mixed audit traces.

#### Scenario: Graph without feeds remains supported
- **WHEN** a real trace contains audit and lifecycle relations but no `feeds` edges and all recorded weights equal 1.0
- **THEN** relation affinity and typed topology still define positive transitions without a dataset-specific fallback

## ADDED Requirements

### Requirement: Logical output dependencies are query-independent
The graph domain SHALL derive logical output-to-output dependencies from native/deterministic provenance structure without reading a query or label. `data.feeds` SHALL collapse producer-output/consumer-call/consumer-output structure, and `resource.flow` SHALL connect the latest prior writer output to a later reader output for one explicit artifact. Temporal adjacency alone MUST NOT create a dependency.

#### Scenario: Feeds dependency is collapsed
- **WHEN** one output feeds a later call and that call returns a target output
- **THEN** the logical view contains a directed source-output to target-output `data.feeds` dependency with supporting physical edge IDs

#### Scenario: Shared artifact creates lifecycle flow
- **WHEN** a call writes an artifact and a later call reads the same artifact
- **THEN** the logical view connects the writer's output to the reader's output using `resource.flow`

#### Scenario: Only temporal adjacency exists
- **WHEN** two calls are consecutive but share no feed or artifact lifecycle
- **THEN** no logical output dependency is created

### Requirement: Provenance-path retrieval uses frozen Dense seeds without labels
The method SHALL rank candidates with the configured frozen Dense encoder, SHALL use only the request query, candidates, and query-independent provenance graph, and MUST NOT inspect query intent, answer IDs, support IDs, dataset identity, or review metadata.

#### Scenario: Hidden intent cannot select traversal direction
- **WHEN** the same graph is queried for upstream and downstream evidence
- **THEN** the method uses the same bounded bidirectional search procedure for both requests

### Requirement: Search is schema-gated and bounded
The method SHALL traverse only logical provenance dependencies and SHALL bound seeds, dependency hops, expansions, partners per anchor, and the protected Dense prefix. Proposal ordering and conflicts SHALL be deterministic.

#### Scenario: Reverse upstream completion
- **WHEN** a Dense seed is a downstream output and its upstream producer is ranked below it
- **THEN** reverse traversal may propose the producer while preserving the dependency's stored source-to-target orientation

#### Scenario: Multi-hop completion
- **WHEN** a seed is connected to relevant output partners by a logical path within the configured hop limit
- **THEN** the bounded closure may promote those partners subject to partner and expansion budgets

#### Scenario: Hop limit is exceeded
- **WHEN** a partner is reachable only beyond the configured maximum logical hops
- **THEN** that partner cannot affect ranking

### Requirement: Interventions preserve Dense ordering invariants
Accepted partners SHALL be stably inserted after their anchor subject to the protected prefix. Non-promoted candidates SHALL retain relative order, and the original descending Dense score multiset SHALL be assigned to the final identifier order.

#### Scenario: No effective proposal
- **WHEN** every proposal is rejected or causes no movement
- **THEN** ranked IDs and scores are exactly Dense and the trace declares exact fallback

#### Scenario: Partner is promoted
- **WHEN** a valid below-anchor partner wins deterministic conflict resolution
- **THEN** it moves after the anchor, protected candidates retain order, and scores remain monotonically descending

### Requirement: Native trace is closed and auditable
The method SHALL emit a closed native trace containing original/final Dense ranks, seeds, logical dependencies, accepted and rejected proposals, path orientation, original/final partner ranks, protected prefix, emitted candidate edges, bounds, and exact fallback state.

#### Scenario: Reverse path emits stored direction
- **WHEN** a proposal traverses a dependency in reverse to find an upstream candidate
- **THEN** the trace records reverse traversal but the emitted logical edge remains source-to-target

#### Scenario: Candidate context is invalid
- **WHEN** a trace dependency, proposal, or emitted edge references an ID outside the request candidates
- **THEN** ranked-result validation fails

### Requirement: Evaluation edges reflect selected logical dependencies
The method SHALL emit only logical dependencies whose endpoints are both present in the returned top-k subgraph and whose path contributed to an accepted intervention. Evaluation edges SHALL use the shared `feeds` category while native trace records retain the namespaced relation.

#### Scenario: Accepted path leaves one endpoint outside top-k
- **WHEN** an accepted path dependency does not have both endpoints in top-k after all insertions
- **THEN** that dependency remains in native diagnostics but is not emitted as a public retrieved edge

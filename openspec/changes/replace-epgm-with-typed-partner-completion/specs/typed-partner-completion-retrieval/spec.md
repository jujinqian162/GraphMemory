## ADDED Requirements

### Requirement: Query-conditioned relation affinity is the only typed signal
The retriever SHALL derive per-edge-type traversal affinity from the request query and frozen schema-owned relation descriptions using the configured frozen encoder, normalized over the edge types present in the graph. It MUST NOT use hand-authored per-type priors, recorded edge weight magnitude, field-binding validity, or lifecycle state to admit or reject a traversal, and MUST NOT inspect dataset identity, answer labels, or gold evidence.

#### Scenario: Revision query prioritizes revision semantics
- **WHEN** a query is more similar to the frozen descriptions of `invalidates` and `supersedes` than to `precedes`
- **THEN** those edge types receive higher traversal affinity, without any dataset name being consulted

#### Scenario: Constant recorded weights do not change traversal
- **WHEN** every edge in the graph has recorded weight 1.0
- **THEN** proposal confidences are identical to the same graph with varied positive recorded weights

#### Scenario: Unbound feeds edge is still traversable
- **WHEN** a `feeds` edge carries no field binding or a binding that does not match its endpoints
- **THEN** the edge remains traversable and its affinity comes only from its edge type

#### Scenario: Affinity is reproducible
- **WHEN** two runs share the query, graph, encoder identity, relation-description version, and method config
- **THEN** relation similarities and affinities are identical

### Requirement: Scope-membership relations are not traversable
The retriever SHALL exclude `contains` edges from partner proposal traversal, because scope membership establishes no evidential dependency between its endpoints and would make every node in an execution scope a bounded-hop neighbour of every other.

#### Scenario: Two outputs in one task scope are not partners
- **WHEN** two candidate outputs are connected only through a shared `task` node by `contains` edges
- **THEN** no partner proposal is generated between them

### Requirement: Bounded typed walk enumerates partner proposals exhaustively
The retriever SHALL enumerate every request candidate reachable from each of the leading `anchor_top_a` Dense candidates within `max_hops` undirected typed steps without revisiting a node. Enumeration MUST NOT depend on the order in which proposals are discovered, and MUST NOT be terminated by a selection budget.

#### Scenario: Direct typed neighbour is proposed
- **WHEN** an anchor has a candidate neighbour one admissible typed edge away
- **THEN** a proposal for that partner exists with a one-step path

#### Scenario: Partner reached through a non-candidate connector is proposed
- **WHEN** two candidate outputs are joined through a non-candidate tool-call node within `max_hops`
- **THEN** a proposal for the far candidate exists and records the connector in its path

#### Scenario: Beyond the hop bound is not proposed
- **WHEN** the shortest admissible typed path between an anchor and a candidate exceeds `max_hops`
- **THEN** no proposal is generated for that pair

#### Scenario: A distant candidate is proposable regardless of nearer candidates
- **WHEN** several candidates lie between an anchor and a further reachable candidate within the hop bound
- **THEN** the further candidate still receives its own proposal, because enumeration is exhaustive rather than budget-consuming

### Requirement: Proposal confidence combines anchor relevance, relation affinity, and path length
The retriever SHALL score each proposal as normalized anchor Dense relevance multiplied by the product of its path edge-type affinities and by `hop_decay` raised to one less than the path length. Confidence MUST be finite and non-negative.

#### Scenario: Weak anchor cannot promote confidently
- **WHEN** two anchors reach equally typed partners over identical paths but one anchor has much lower Dense relevance
- **THEN** the lower-relevance anchor produces the lower proposal confidence

#### Scenario: Longer path scores lower
- **WHEN** the same partner is reachable by a one-step and a two-step admissible path
- **THEN** the one-step path yields the higher confidence and is the retained proposal

### Requirement: Accepted promotions are one-to-one and leave the protected prefix intact
The retriever SHALL accept at most one partner per anchor and at most one anchor per partner, SHALL reject any proposal below `min_partner_confidence`, and SHALL reject any partner whose Dense rank is within `preserve_dense_top_n` or is not strictly worse than its anchor's Dense rank. Every rejected proposal SHALL carry a machine-readable reason.

#### Scenario: One anchor promotes one partner
- **WHEN** one anchor has several accepted-eligible partners
- **THEN** only the highest-confidence partner is promoted and the others are rejected as lower confidence for that anchor

#### Scenario: Protected prefix is never displaced
- **WHEN** a proposal targets a candidate inside the Dense top-`preserve_dense_top_n`
- **THEN** the proposal is rejected and the prefix keeps its Dense order

#### Scenario: Upward-only promotion
- **WHEN** a proposal's partner already ranks better than its anchor
- **THEN** the proposal is rejected because promotion would not improve the partner's position

#### Scenario: Rejections are auditable
- **WHEN** any proposal is not accepted
- **THEN** the trace records a non-empty reason and accepted proposals record no reason

### Requirement: Promotion reorders within the ranked head and preserves the score multiset
The retriever SHALL move each accepted partner to the position immediately after its anchor, no earlier than `preserve_dense_top_n`. The operation SHALL be a permutation of candidate ids over the unchanged ordered Dense score slots, so that intra-top-`k` order MAY change while the returned score multiset is exactly the Dense score multiset.

#### Scenario: Partner enters the top-5 behind a top-3 anchor
- **WHEN** an accepted partner has Dense rank 8 and its anchor has Dense rank 3
- **THEN** the partner occupies rank 4 in the returned ranking

#### Scenario: Score multiset is preserved exactly
- **WHEN** any promotion is applied
- **THEN** the returned scores are the Dense scores in their original descending slot order

#### Scenario: Promotion that cannot move the partner is rejected
- **WHEN** the computed insertion position is not better than the partner's current position
- **THEN** the proposal is rejected as having no effective insertion

### Requirement: Connectors never occupy ranked positions
Non-candidate native graph nodes SHALL serve only as interior path evidence. They MUST NOT appear in the ranked node list or in the shared retrieved-subgraph node surface, and MUST be reported in the native trace.

#### Scenario: Tool call joins two candidate outputs
- **WHEN** an accepted proposal's path passes through a non-candidate tool call
- **THEN** the ranked list contains only request candidates and the trace records the tool call as a connector

### Requirement: Emitted candidate edges are oriented by stored direction
The retriever SHALL emit one candidate-level dependency edge per accepted promotion, oriented along the stored provenance direction. A path whose steps do not agree on a single traversal direction SHALL emit no candidate edge while still permitting the promotion.

#### Scenario: Reverse walk emits stored orientation
- **WHEN** an accepted partner is reached by traversing a stored edge against its direction
- **THEN** the emitted candidate edge follows the stored source-to-target orientation

#### Scenario: Divergent path emits no fabricated dependency
- **WHEN** an accepted path leaves a shared upstream node toward both endpoints
- **THEN** the promotion applies and no candidate dependency edge is emitted

### Requirement: Exact Dense fallback is explicit
The retriever SHALL return the Dense ranking and scores unchanged when no proposal is accepted, and the native trace SHALL declare the fallback so that a no-op is directly observable rather than inferred from metric equality.

#### Scenario: Edgeless candidate graph
- **WHEN** no admissible typed path connects any anchor to another candidate
- **THEN** ranked nodes and scores are exactly Dense and the trace declares exact fallback with no accepted proposals

#### Scenario: Fallback state is consistent
- **WHEN** the trace declares exact fallback
- **THEN** it contains no accepted proposal, no connector, and no emitted edge

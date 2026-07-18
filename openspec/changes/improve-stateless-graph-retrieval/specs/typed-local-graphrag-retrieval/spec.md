## ADDED Requirements

### Requirement: GraphRAG request owns immutable input graph evidence
`GraphRAGRequest` SHALL represent immutable typed entity mentions and title-entity groups. Runtime resolver evidence and directed candidate bridge proposals SHALL be method-native trace outputs. The request SHALL remain distinct from execution-provenance and evidence-graph request contracts.

#### Scenario: Title and body mentions coexist
- **WHEN** a candidate title names one entity and its body mentions another
- **THEN** the request distinguishes `TITLE_ENTITY` from `MENTIONS` with candidate/entity IDs, normalized surface, priors, document frequency, and mention confidence

### Requirement: Title groups do not fan out candidate bridges
A title entity group SHALL contain all candidates sharing a stable normalized title entity and group statistics, but SHALL be resolver input only. A body mention SHALL NOT directly produce bridges to every group member.

#### Scenario: Multiple sentences share a title
- **WHEN** an anchor mentions an entity whose title group has several candidates
- **THEN** no candidate bridge exists until the sentence resolver selects at most one target

### Requirement: Entity evidence is deterministic and hub controlled
Entity normalization, source/alias priors, normalized IDF, and mention confidence SHALL be deterministic and auditable. Ambiguous aliases and entities above `max_entity_document_frequency_ratio` SHALL NOT produce an accepted bridge. `SHARED_ENTITY` alone SHALL be diagnostic-only.

#### Scenario: High-frequency entity connects candidates
- **WHEN** an entity's document-frequency ratio exceeds the configured limit
- **THEN** its mentions can be traced but cannot trigger promotion

### Requirement: Frozen Dense sentence resolution is batched and abstaining
The GraphRAG-private resolver SHALL reuse the same Frozen Dense encoder instance as the Dense ranker, batch unique passages and bounded source-aware anchor queries, rank group candidates by `(-score, original_dense_rank, node_id)`, select top-1 only, and abstain on multi-candidate groups below `min_sentence_score_margin`.

#### Scenario: Singleton group resolves
- **WHEN** a stable title group contains one non-anchor candidate
- **THEN** that candidate is resolved deterministically as an unambiguous singleton

#### Scenario: Multi-candidate group has low margin
- **WHEN** top-1 minus top-2 score is below the configured margin
- **THEN** resolution rejects with `ambiguous_title_sentence` and produces no bridge

#### Scenario: Resolver scores tie
- **WHEN** two group candidates have equal normalized cosine score
- **THEN** original Dense rank and then node ID select the unique top-1 target

### Requirement: Bridge confidence is pair-local and non-duplicative
The bridge confidence SHALL be the geometric mean of seed confidence, source body-mention confidence, and target title-mention confidence. Resolver margin SHALL be a prior hard gate and SHALL NOT be multiplied into final confidence. Multiple entity proofs for one pair SHALL retain the highest-confidence proof and trace the others as rejected evidence.

#### Scenario: Bridge survives all gates
- **WHEN** a Dense top-S anchor has a valid low-frequency mention, a uniquely resolved title candidate, and confidence above threshold
- **THEN** exactly one directed `BRIDGE_TO` proposal is created for that anchor/group

### Requirement: GraphRAG performs only local stable insertion
GraphRAG SHALL originate proposals only from original Dense top-S anchors, accept at most one partner per anchor across groups, resolve partner conflicts deterministically, preserve the protected prefix, and preserve relative order of all non-promoted candidates. It SHALL NOT run PPR, RRF, or global graph-score fusion.

#### Scenario: One bridge is accepted
- **WHEN** a unique bridge partner lies below its anchor and outside the protected prefix
- **THEN** only that partner is stably inserted after the anchor subject to the protected prefix

### Requirement: No accepted bridge returns exact Dense output
When no bridge causes insertion, GraphRAG SHALL return the original Dense node order, scores, and `RankedNode` objects unchanged.

#### Scenario: Only shared or rejected entities exist
- **WHEN** all graph evidence is `SHARED_ENTITY`, hub-suppressed, ambiguous, below confidence, or a no-op
- **THEN** GraphRAG is exactly Dense and trace marks fallback

### Requirement: GraphRAG emits only promoted candidate bridges
Only an accepted directed `BRIDGE_TO` whose partner actually moved and whose endpoints are present in final top-k SHALL appear in retrieved edges. Title groups, entity nodes, shared entities, rejected proposals, and no-ops SHALL remain native trace only.

#### Scenario: Resolver selects one of many title sentences
- **WHEN** that selected target is promoted
- **THEN** exactly one candidate-level edge is emitted rather than an edge to every same-title sentence

### Requirement: GraphRAG configuration is strict and non-training
Configuration SHALL expose `seed_top_s`, `max_entity_document_frequency_ratio`, `sentence_resolver=frozen_dense`, `min_sentence_score_margin`, `min_bridge_confidence`, `max_partners_per_anchor=1`, and `preserve_dense_top_n`. Retired PPR/fusion fields and alternate/fallback resolvers SHALL be rejected.

#### Scenario: Global graph field is configured
- **WHEN** `restart_probability`, `max_iterations`, `semantic_weight`, `entity_weight`, or an unsupported resolver is supplied
- **THEN** strict configuration loading fails

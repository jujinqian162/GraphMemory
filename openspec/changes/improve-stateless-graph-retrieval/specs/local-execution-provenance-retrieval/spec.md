## ADDED Requirements

### Requirement: Structural signals are validity gates
The execution-provenance retriever SHALL accept a path only if it starts at an original Dense seed, ends at a distinct candidate, has endpoint-consistent binding, contains the required `FEEDS` and `RETURNS` structure, and is not invalidated. Binding, completeness, grounding, and lifecycle SHALL NOT be additive score terms.

#### Scenario: Invalid path is rejected
- **WHEN** binding is mismatched, a required structural edge is absent, or the path is invalidated
- **THEN** the path cannot produce a ranking proposal regardless of its Dense endpoint score

### Requirement: Path confidence uses existing semantic edge weight once
The retriever SHALL compute path confidence from semantic edge weights already present in the existing request graph, multiplied only by the configured hop penalty beyond two edges. Semantic rank, raw semantic score, base-query Dense scores, and fixed structural bonuses SHALL NOT be multiplied or added again. The retriever SHALL NOT regenerate or migrate dataset graph data.

#### Scenario: Initial two-hop path is scored
- **WHEN** a valid two-hop synthetic path contains one semantic `FEEDS` edge
- **THEN** path confidence equals that edge's existing request weight

#### Scenario: Low-confidence graph abstains
- **WHEN** existing semantic edge weights are below `min_path_confidence`
- **THEN** no path proposal is accepted

### Requirement: Search is bounded per original Dense seed
Only original Dense top `seed_top_s` candidates SHALL originate proposals. Search SHALL obey `beam_width`, `max_hops`, `max_path_expansions`, and `max_paths_per_seed=1`, deduplicate equivalent candidate edges, and return at most one partner for each seed.

#### Scenario: Multiple valid paths share a seed
- **WHEN** a seed reaches multiple valid partners or duplicate logical paths
- **THEN** only the deterministic highest-confidence partner proposal remains for that seed

### Requirement: Proposal conflicts are deterministic
If multiple anchors propose the same partner, the retriever SHALL retain highest confidence and break ties by original anchor Dense rank, original partner Dense rank, and node ID.

#### Scenario: Same partner has two anchors
- **WHEN** two accepted path proposals target one candidate
- **THEN** conflict resolution returns exactly one proposal using the declared ordering without consulting a mutated ranking

### Requirement: Local insertion preserves Dense invariants
The first `preserve_dense_top_n` Dense candidates SHALL retain members and order. Accepted partners MAY move only to immediately after their anchor subject to the protected prefix. Every non-promoted candidate SHALL preserve Dense relative order.

#### Scenario: Partner is promoted after protected prefix
- **WHEN** a valid proposal targets a candidate below its anchor and outside the protected prefix
- **THEN** only that partner moves to `max(protected_end + 1, anchor_final_position + 1)` and other candidates remain stable

#### Scenario: Proposal would move an earlier node
- **WHEN** a partner already precedes its anchor or belongs to the protected prefix
- **THEN** the proposal is a no-op and emits no retrieved edge

### Requirement: No effective path returns exact Dense output
When no proposal causes an insertion, the retriever SHALL return the original Dense node order, scores, and `RankedNode` objects without recomputation.

#### Scenario: All paths are rejected
- **WHEN** every path fails validity or confidence gates
- **THEN** node order and score values are exactly equal to Dense output and trace marks exact fallback

### Requirement: Output edges correspond to real promotions
The retriever SHALL emit only directed candidate-level edges from accepted paths that actually promoted a partner and whose endpoints appear in final top-k. Rejected, structural-only, duplicate, and no-op paths SHALL remain trace-only.

#### Scenario: Accepted path changes top-k
- **WHEN** stable insertion promotes a partner into the evaluated result
- **THEN** exactly its directed candidate edge is emitted with bounded edge count

### Requirement: Configuration is strict and current-only
The method config SHALL expose only `seed_top_s`, `beam_width`, `max_hops`, `max_paths_per_seed`, `max_path_expansions`, `min_path_confidence`, `preserve_dense_top_n`, and `hop_penalty` with strict validation. Retired global-bonus fields SHALL be unknown-field errors.

#### Scenario: Retired path weight is configured
- **WHEN** `semantic_weight`, `binding_weight`, `top_paths`, or another retired field is supplied
- **THEN** configuration loading fails rather than translating or ignoring it

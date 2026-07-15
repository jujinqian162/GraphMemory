## MODIFIED Requirements

### Requirement: Semantic seed selection
Retriever SHALL score every retrievable execution candidate once, select a configurable `seed_top_s` set whose effective size is at least the requested ranking budget when candidates permit, and exclude Task/Agent connector nodes from seed competition.

#### Scenario: Requested budget exceeds configured seeds
- **WHEN** `top_k` is 10, configured `seed_top_s` is 3, and at least 10 candidates exist
- **THEN** the effective seed pool contains at least 10 candidates and the final ranking can return 10 distinct candidates

### Requirement: Typed dependency expansion
Retriever SHALL expand legal directed dependency transitions incrementally, MUST preserve edge direction and FieldBinding metadata, and MUST NOT insert unconditional reverse adjacency; reverse context is traversable only through an explicitly supported reverse transition.

#### Scenario: Follow field binding
- **WHEN** ToolOutput feeds a downstream ToolCall through a valid output-field/input-parameter binding and that call returns a ToolOutput
- **THEN** expansion can reach the downstream output and records both native typed edges

#### Scenario: Reject merely present but inconsistent binding
- **WHEN** a `feeds` edge has a non-null binding whose value hash does not match the declared source output field or whose input parameter is not declared by the target call
- **THEN** that edge receives zero binding-consistency credit and validation rejects generated benchmark records containing it

#### Scenario: Incoming edge only
- **WHEN** the current node has only an incoming dependency edge and no supported reverse transition is configured
- **THEN** expansion does not traverse that edge backwards

### Requirement: Provenance path scoring
Retriever SHALL use bounded beam search that scores and prunes partial hypotheses at each hop, combining semantic endpoint, path mean and bottleneck relevance, transition completeness, binding consistency, grounding, invalidation, and normalized length; final candidate scores SHALL merge semantic and path evidence rather than replacing semantic scores.

#### Scenario: Incremental pruning
- **WHEN** one expansion step creates more hypotheses than `beam_width`
- **THEN** only the best bounded hypotheses continue to the next hop and unexpanded hypotheses do not re-enter after full-tree enumeration

#### Scenario: Coherent path competes with isolated seed
- **WHEN** a coherent complete path has moderately relevant nodes and an isolated seed has one high semantic score
- **THEN** both semantic and path signals remain represented in final scoring without using only the maximum node score

### Requirement: Trace records actual search
Retriever SHALL return a full ranked candidate list, the actual selected native typed paths and traversed edges, and contracted logical candidate dependency edges for retrieved-subgraph path metrics; MUST NOT use all edges induced by top-k nodes as a substitute for search traces.

#### Scenario: Contract output dependency
- **WHEN** the selected path traverses `output_a -> feeds -> call_b -> returns -> output_b`
- **THEN** native trace contains the two typed edges and the retrieved logical subgraph contains candidate edge `output_a -> output_b`

#### Scenario: Honor requested top-k
- **WHEN** at least `top_k` candidates exist
- **THEN** the public selected node list contains exactly `top_k` distinct candidate IDs

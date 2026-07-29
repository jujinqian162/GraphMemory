## ADDED Requirements

### Requirement: R-GCN weight removal preserves graph and message mass
Full provenance R-GCN tensorization SHALL consume schema-v3 artifact weights. `wo_edge_weight` SHALL preserve nodes, endpoints, forward/reverse directions, relation IDs, bindings, and per-source total feed weight while replacing within-source feed weights with their arithmetic mean; non-feed weights SHALL remain unchanged.

#### Scenario: Weighted source is tensorized by the full model
- **WHEN** one source has artifact feed weights `0.9` and `0.6`
- **THEN** its forward and reverse message edges retain `0.9` and `0.6`

#### Scenario: The same source is tensorized without edge weights
- **WHEN** the request is tensorized under `wo_edge_weight`
- **THEN** both feed edges use `0.75`, total source feed mass remains `1.5`, and all non-weight graph tensors equal the full-model tensors

### Requirement: Provenance pair construction consumes native logical competitors
The pair stage SHALL use a provenance-owned typed build task containing the text request, execution-provenance graph, and evidence label. It SHALL materialize `hard_provenance_successor` negatives from non-gold successors of gold sources and `hard_provenance_predecessor` negatives from non-gold sources participating in competing transitions into gold targets, in addition to configured dense, BM25, and easy negatives.

#### Scenario: Gold source has competing successors
- **WHEN** a gold source has one gold and multiple non-gold logical successors
- **THEN** configured non-gold successors are materialized as provenance-successor negatives before generic semantic negatives

#### Scenario: Non-provenance method builds pairs
- **WHEN** a flat or EvidenceGraph method executes the pair stage
- **THEN** it continues to use its existing typed pair task and does not receive an execution-provenance graph

### Requirement: Negative candidates are unique with explicit hardness precedence
Each candidate SHALL appear at most once per task's pair artifact. If multiple samplers select one candidate, the retained type SHALL follow `hard_provenance_successor`, `hard_provenance_predecessor`, `hard_dense`, `hard_bm25`, then `easy_random`, and the summary SHALL report source overlaps and post-dedup counts.

#### Scenario: Dense and provenance samplers select the same candidate
- **WHEN** one candidate is a competing gold-source successor and also a dense hard negative
- **THEN** exactly one negative pair is emitted with type `hard_provenance_successor` and the overlap is counted in the pair summary

#### Scenario: Requested count exceeds unique candidates
- **WHEN** configured samplers request more negatives than the candidate pool provides
- **THEN** the stage emits every eligible candidate at most once and records the shortfall instead of duplicating logits in the loss

### Requirement: Candidate ranking loss is task-balanced and pairwise
Training SHALL compare every selected positive with every unique selected negative using pairwise logistic loss, average comparisons within each task, and then average tasks. Candidate loss, class-balanced logical-edge loss, total loss, and pair-category counts SHALL be logged separately.

#### Scenario: One task has more hard negatives
- **WHEN** two tasks have different numbers of unique negatives
- **THEN** each task contributes equal outer weight to candidate loss after its internal comparisons are averaged

#### Scenario: Both gold outputs are supervised
- **WHEN** a task has two gold outputs and at least one negative
- **THEN** both gold logits are independently compared against every selected negative

### Requirement: Hard-negative ablation removes only hard candidate supervision
`wo_hard_negatives` SHALL retain positive and easy-random pairs, set provenance, dense, and BM25 hard-negative counts to zero, and preserve the full logical-edge supervision contract. It SHALL own variant-local pair, model, prediction, and evaluation artifacts.

#### Scenario: Hard-negative variant is planned
- **WHEN** `method.variant=wo_hard_negatives` is selected
- **THEN** workflow invalidation begins at pairs and the pair summary contains no hard-negative sample type

#### Scenario: Edge loss is computed in the hard-negative variant
- **WHEN** a task contains legal non-gold logical transitions
- **THEN** those transitions remain negative edge targets even though hard candidate pairs were removed

### Requirement: Learned edges can perform bounded stable candidate promotion
Full inference SHALL begin from the complete raw candidate-logit ranking, consider only configured top-M candidates and seed sources, accept only above-threshold logical transitions, retain at most one successor per seed, resolve target conflicts deterministically, and promote an eligible lower-ranked target directly after its source while preserving the protected prefix and all non-promoted relative order.

#### Scenario: High-confidence partner completes a candidate set
- **WHEN** an eligible target is below its selected source, outside the protected prefix, and its edge probability meets the threshold
- **THEN** the target is stably inserted after the source and the trace records original rank, final rank, edge probability, and responsible transition

#### Scenario: Target already precedes its source
- **WHEN** an accepted edge points to a candidate already above the source
- **THEN** candidate order is unchanged and the transition is recorded as a no-op

#### Scenario: Two sources propose the same target
- **WHEN** multiple eligible transitions target one candidate
- **THEN** exactly one promotion survives using edge probability, original source rank, original target rank, and target ID

### Requirement: Edge reranking has a truthful rank-stage ablation
The public provenance suite SHALL expose `wo_edge_rerank` in addition to `wo_graph`, `wo_edge_type`, `wo_edge_weight`, and `wo_hard_negatives`. `wo_edge_rerank` SHALL reuse the full trained checkpoint and pair artifact, skip structured promotion, and preserve the raw candidate-logit order while retaining edge-head diagnostics.

#### Scenario: Edge-rerank variant is selected
- **WHEN** `method.variant=wo_edge_rerank` is planned
- **THEN** pairs and model artifacts alias the full variant while ranking and evaluation own distinct variant artifacts

#### Scenario: Raw and structured orders differ
- **WHEN** full inference performs a valid promotion
- **THEN** `wo_edge_rerank` returns the unpromoted raw order and its trace identifies that promotion was disabled

### Requirement: Logical-edge inference can abstain
Inference SHALL emit a logical dependency only when its probability meets the configured threshold, both endpoints appear in final top-k, and it is the retained successor for its source. It MUST NOT emit a successor solely because a selected source has candidates.

#### Scenario: All successor logits are low
- **WHEN** every successor probability for a selected source is below threshold
- **THEN** no edge is emitted for that source and the trace records abstention

#### Scenario: Accepted target is outside final top-k
- **WHEN** an above-threshold transition has an endpoint outside final top-k
- **THEN** it is not included in retrieved logical or native edges

### Requirement: Provenance checkpoints describe the effective structured model
Schema-v3 checkpoints SHALL store dataset/construction identity, encoder identity, node/relation vocabularies, edge-weight policy, pair-sampling policy, candidate-loss type, loss weights, structured-inference settings, selection-objective definition, effective variant, and best-epoch component metrics. Pre-v3 provenance and evidence-R-GCN checkpoints MUST be rejected.

#### Scenario: V2 provenance checkpoint is loaded
- **WHEN** ranking requests a v2 checkpoint under the v3 method
- **THEN** loading fails before tensorization with an explicit schema error

#### Scenario: Variant checkpoint is loaded
- **WHEN** a model-changing ablation checkpoint is used
- **THEN** its stored effective policy exactly matches the selected variant and incompatible policy overrides fail

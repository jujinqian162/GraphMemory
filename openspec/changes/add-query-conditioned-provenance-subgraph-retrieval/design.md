## Context

The non-trained EPGM currently takes the frozen Dense ranking, expands bounded independent paths from the top seeds, and fuses each target's best path back into a node ranking. This architecture has three measured failure modes:

1. graph nodes outside the Dense seed neighborhood cannot become candidates even when they are one provenance hop from relevant evidence;
2. independent target paths do not optimize complete support or emitted-edge coherence;
3. recorded `feeds` weights are source-local successor probabilities, but global multiplicative `w_eff` treats them as globally comparable path confidence. A read-only 2,851-task RQ2 ablation reproduced the stored run exactly and showed that `w_eff` reduced Path Recall@10 from 18.48% to 11.15% and gold-edge true positives from 505 to 310.

RQ2 graphs contain calibrated field-bound `feeds`/`returns` structure; RQ3 real traces contain mixed audit, revision, delegation, and lifecycle edges whose recorded weights are constant placeholders. The replacement must use one schema-driven algorithm without dataset identity, gold labels, task-specific training, or regenerated graphs.

## Goals / Non-Goals

**Goals:**

- Make the full native provenance graph a query-conditioned candidate source.
- Interpret recorded confidence only in the source-local context in which it was generated.
- Use frozen encoder semantics to condition relation traversal on the query.
- Jointly select a compact connected evidence subgraph under the retrieval `top_k` budget.
- Permit task, agent, and tool-call nodes as connector nodes without exposing them as ranked evidence unless they are request candidates.
- Preserve deterministic output, contract validation, cache reproducibility, oriented logical edges, and exact Dense fallback.
- Keep one public registry method id and one default non-trained architecture across RQ2 and RQ3.

**Non-Goals:**

- Train relation weights, a query router, or any new encoder.
- Hand-assign RQ3 edge weights or branch on dataset names.
- Implement an exact NP-hard directed PCST solver or add a mandatory native dependency.
- Change evidence/path evaluation definitions, dataset converters, trainable EPGM R-GCN, or candidate text construction.
- Guarantee benchmark improvement without the server experiment; implementation correctness and invariant tests precede empirical claims.

## Decisions

### 1. Add one `ppr_steiner` default and retain legacy modes only as diagnostics

`EpgmRetrieverConfig.variant` gains `ppr_steiner` and makes it the default. `typed_beam` and `dependency_path` remain frozen diagnostic variants during migration so historical results and causal ablations remain reproducible. The public registry id stays `execution_provenance_retriever`.

Alternative: mutate `typed_beam` in place. Rejected because historical cache identity and paper comparisons would become ambiguous.

### 2. Relation semantics are schema-owned and query-conditioned with the frozen encoder

Every public `ProvenanceEdgeType` has a stable natural-language description owned by the EPGM method. The existing frozen Dense encoder embeds these descriptions once per retriever and embeds each query. Cosine similarity is converted to a positive relation affinity by a temperature-scaled softmax over relation types present in the request graph.

No query labels, answer text, dataset id, or benchmark-specific keywords are available to this calculation. Relation descriptions are versioned as part of the ranking/cache identity.

Alternative: fixed type priors only. Rejected because the same relation should not receive the same traversal preference for lifecycle, delegation, and dependency questions.

### 3. Transition confidence is normalized per source, never globally

Each stored edge becomes a forward arc and, when allowed, a reverse arc with a fixed reverse penalty. Before normalization, arc strength is:

```
strength(e, q) =
    max(type_prior(type(e)), min_edge_prior)
  * relation_affinity(type(e), q)
  * recorded_confidence(e)
  * direction_factor(e)
  * hub_factor(e)
```

`recorded_confidence(e)` is the raw finite edge weight in the stored forward direction. Because outgoing strengths are normalized independently for each source node, RQ2's source-local `feeds` probabilities retain their intended semantics. A non-zero reverse arc treats that confidence as topology eligibility plus the fixed reverse penalty rather than comparing unrelated incoming-source probabilities; recorded zero disables both directions. Constant `1.0` weights in RQ3 are an exact identity factor. No variance detector, min-max transform, or dataset switch remains.

Invalid field-bound transitions are excluded. High-degree task/agent/`contains` traversal is controlled with deterministic degree normalization rather than a global ban, allowing session-structure questions without making hubs free shortcuts.

Alternative: direct path multiplication. Rejected by the measured RQ2 regression and because source-local probabilities are not globally comparable.

### 4. Typed Personalized PageRank performs candidate expansion

Dense relevance over all request candidates is min-max scaled with a fixed positive numerical floor and normalized into teleport mass before seeding PPR over every native graph node. This preserves the narrow cosine-score dynamic range that a unit-temperature softmax would otherwise flatten. Iteration uses fixed damping, L1 tolerance, and a maximum iteration count. Dangling mass returns to the teleport distribution. Transition ordering and floating-point accumulation order are stable.

PPR produces graph relevance for candidate and connector nodes; it does not itself decide the returned evidence set.

Alternative: larger beam or k-shortest paths. Rejected because both remain dependent on early path pruning and independently selected endpoints.

### 5. Use a deterministic budgeted Steiner-style greedy extractor

The implementation uses a solver-free rooted connected-subgraph approximation suitable for the small request graphs:

1. assign candidate prizes from normalized Dense relevance and candidate PPR mass minus a fixed candidate-inclusion cost;
2. choose the highest-prize candidate as the initial root;
3. repeatedly compute deterministic residual minimum-cost paths from the selected evidence anchors to unselected candidates using `-log(transition_probability) + connector_hop_cost` for new arcs and zero marginal cost for arcs already in the selected tree; this exposes the current connector frontier without losing a candidate anchor or auditable full path;
4. evaluate every path against the Dense top-`k` incumbent. A candidate already in the incumbent pays no displacement cost; an out-of-budget candidate pays the prize of the weakest non-selected incumbent evidence it would replace;
5. add the path with the largest positive marginal `candidate_prize - new_connector_cost - displaced_dense_prize`, including any candidate nodes on that path, while the connected evidence-node count stays within `top_k`;
6. connector-only graph nodes do not consume the evidence budget;
7. stop when no positive counterfactual marginal remains.

The accepted membership changes are projected onto the Dense ranking while retaining original Dense-relative order within both the final top-`k` and tail. This lets structure complete a support set without moving graph-central evidence ahead of stronger semantic hits. The reordered ids occupy the original descending Dense score slots so the full score multiset remains stable and every method returns the full candidate list.

If the extractor selects fewer than two candidates, has no positive-marginal connection step, or fails any numerical/connectivity invariant, the result is byte-for-byte Dense identity. A structural `contains`/`invokes` connection may alter ranking while emitting no shared logical dependency edge; only native relations covered by the dependency-evaluation contract are collapsed publicly.

Alternative: exact directed PCST. Rejected for the initial implementation because it adds solver complexity without being necessary at graph sizes of roughly 66--229 nodes. The typed trace exposes the objective so an exact solver can replace the approximation later without changing contracts.

### 6. Preserve native direction while allowing penalized reverse traversal

Search arcs may run backward, but every selected edge records the stored graph orientation. Candidate-level collapsed edges are oriented by the stored path direction and use contract edge type `feeds`; the native trace retains the actual provenance relation and traversal direction. Duplicate collapsed endpoint pairs are rejected deterministically.

### 7. Introduce a dedicated closed native trace kind

A new `execution_provenance_subgraph` trace records:

- active variant and relation-description version;
- Dense ranks and teleport mass;
- per-relation query similarity/affinity;
- normalized typed transitions, direction, recorded confidence, and arc cost;
- PPR convergence status, iterations, residual, and final node mass;
- selected evidence nodes, connector nodes, selected stored edges, prizes, new-edge costs, displaced Dense candidates/costs, marginal gains, and total objective;
- emitted candidate edges, scorer identity, and exact Dense fallback.

Validation rejects unknown fields/kinds, duplicate records, non-finite values, transition rows that do not normalize per source, unknown graph references, evidence-budget overflow, disconnected selected subgraphs, invalid orientation, and fallback/intervention inconsistency.

### 8. Cache identity includes every behavior-affecting frozen parameter

The resolved method config and cache key include variant, relation-description version, PPR damping/tolerance/iterations, relation temperature, direction and hub factors, prize weights, and selection threshold. Encoder identity remains part of ranking provenance. No result can reuse a cache generated by the legacy beam implementation.

## Risks / Trade-offs

- **[Frozen relation embeddings add encoder work]** → Cache relation embeddings per retriever and reuse the already loaded Dense encoder; only one query embedding is added per request in the first implementation.
- **[PPR can diffuse through task/agent hubs]** → Degree-normalized hub factors, query-conditioned relation affinity, and positive-marginal connected extraction; test explicit no-hub-shortcut cases.
- **[Greedy Steiner approximation can miss the global optimum]** → Keep deterministic objective traces, tiny exhaustive-oracle tests, and a solver-independent selector interface.
- **[Connected selection can lower early-rank MRR]** → Dense-relative order is retained outside selected promotions and fallback is exact when no auditable connected gain exists; report MRR alongside Full Support/Path/Edge.
- **[Relation descriptions look like hand-written benchmark rules]** → Derive one description per public schema relation, version them, freeze before formal runs, and forbid dataset/query-type branches.
- **[Large transition traces increase artifact size]** → Record only graph arcs with non-zero transition probability and final PPR state; graphs are bounded and current traces already store traversed edges/paths.
- **[Legacy variant surface complicates the paper]** → Public reporting names only `ppr_steiner` as EPGM; legacy variants are diagnostic implementation checks, not peer methods.

## Migration Plan

1. Add algorithm modules and focused invariants without changing the default.
2. Add the closed trace contract, serializer, validator, and round-trip tests.
3. Integrate `ppr_steiner` into `EpgmRetriever`, expose it through existing registry/builders/config, and make it default.
4. Set global `weight_aware` path multiplication off for legacy typed-beam diagnostics.
5. Run local unit/full validation and deterministic smoke graphs; do not claim quality from tests.
6. Run RQ2 seed-13 and RQ3 on the server. Keep legacy artifacts available for paired comparison.
7. Roll back by selecting the frozen `typed_beam` or `dependency_path` diagnostic variant; cache keys remain distinct.

## Open Questions

- The first server run will determine whether the greedy selector's fixed objective weights need to be frozen from RQ2 dev only. RQ3 remains test-only and MUST NOT be used for tuning.
- If the greedy selector is materially below its exhaustive oracle on tiny tests or formal graphs, a later change may introduce an exact/approximate directed PCST backend behind the same selector contract.

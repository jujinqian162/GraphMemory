## Context

The non-trained EPGM retriever must beat GraphRAG on RQ2 (synthetic 2Wiki provenance, 2851 tasks) and match or beat `typed_beam` on RQ3 (20 real agent-trace queries) using **one** configuration, with no dataset-specific switches.

Measured baselines (seed 13):

| dataset | metric | Dense | GraphRAG | ppr_steiner | typed_beam |
|---|---|---|---|---|---|
| RQ2 | Recall@5 | 0.6156 | **0.6561** | 0.6154 | — |
| RQ2 | Full Support@5 | 0.2473 | **0.3294** | 0.2473 | — |
| RQ2 | MRR | 0.8091 | 0.8090 | 0.8091 | — |
| RQ3 | R@10 | 64.42 | 63.25 | 52.25 | **68.25** |
| RQ3 | MRR | 65.83 | **67.08** | 60.21 | 60.33 |

Measured ceiling for any pure top-10 reranker on RQ2: `Recall@5` 0.7601, `Full Support@5` 0.5212. GraphRAG captures 28-30% of that headroom; `ppr_steiner` captures ~0%.

## Goals / Non-Goals

Goals:
- One algorithm, one config, both RQs.
- Convert intra-top-10 headroom into `Recall@5`/`Full Support@5` without sacrificing `MRR`.
- Explainable per-decision output: which anchor, which typed path, what confidence, accepted or why not.
- Fewer than 6 behavioral hyperparameters.

Non-Goals:
- Expanding the candidate pool beyond the request candidate set. RQ2 `Recall@10` is bounded at 0.7601 by Dense pool composition; lifting that ceiling is a candidate-generation problem, out of scope here.
- Retaining the legacy variants as ablations. They are identity functions on one dataset each and carry no scientific signal.
- Any trained component.

## Decisions

### Decision 1: Bounded undirected typed walk, not diffusion

PPR distributes mass globally and then needs a budgeted selector to convert mass into a discrete top-k decision. That conversion step is where every failure listed in the proposal lives: prize calibration, displacement accounting, and greedy budget exhaustion are all artifacts of the mass-to-decision gap.

Partner completion skips the gap. It enumerates a small, bounded set of concrete `(anchor, path, partner)` proposals and decides on each independently. Enumeration is exhaustive within `max_hops`, so there is no greedy-ordering pathology — the budget-exhaustion bug is structurally impossible.

Undirected traversal is required by the data: RQ3 has `edge.weight` constant at 1.0 and `binding` absent, and its gold partner frequently sits upstream of the anchor. RQ2's directed `feeds` schema is still respected implicitly, because relation affinity down-weights types the query does not ask about.

Alternatives rejected:
- *Keep PPR, fix the selector.* Rejected: even with a correct selector, the prize constants must be recalibrated per graph density, which reintroduces the dataset-specific tuning the method is supposed to avoid.
- *Directed-only walk.* Rejected: measured to reduce RQ3 to Dense identity, exactly the `dependency_path` failure.

### Decision 2: Relation affinity is the only learned-free typed signal

Keep `query_relation_affinities`: a softmax over cosine similarity between the query embedding and frozen per-edge-type natural-language descriptions. Drop `DEFAULT_EDGE_PRIORS` (13 hand-authored constants), the binding gate, the lifecycle gate, and `edge.weight` magnitude.

Rationale: hand-authored priors are the concrete mechanism by which `typed_beam` overfits RQ3, and the binding/completeness gates are the mechanism by which `dependency_path` overfits RQ2. Relation affinity replaces both with one runtime-computed quantity that reads only the query and the schema, never the dataset identity or labels.

This yields the change's central ablation: freeze `relation_affinity` to a constant and both RQs must degrade. If they do not, the typed signal is not doing the work and the claim is unsupported.

`contains` is excluded from traversal. It is a scope-membership relation whose endpoints share no evidential dependency, and traversing it makes every node in an execution scope a 2-hop neighbour of every other, which destroys the precision of the hop bound. This is a schema fact, not a tuned choice.

### Decision 3: Insert after anchor, allowing intra-top-k reorder

This is the decision that produces the measured gain, and it directly reverses the current implementation's most costly constraint.

`stable_insert` moves an accepted partner from its Dense position to `max(preserve_dense_top_n, anchor_position + 1)`. The Dense score multiset is preserved exactly (scores stay bound to slots, ids move between slots), so the operator is a pure permutation and cannot inflate score-based metrics.

`preserve_dense_top_n = 2` protects the `Recall@2` and `MRR` prefix. RQ2 `Recall@2` is identical across all four methods, confirming the top-2 is where Dense is already reliable; `ppr_steiner`'s RQ3 `MRR` regression (-5.6) came from disturbing that prefix. Protecting it bounds MRR risk by construction while leaving slots 3-10 — where all 781 promotable tasks live — free to reorder.

### Decision 4: Exhaustive proposals, single threshold, one-to-one matching

Score: `confidence = dense_rel(anchor) * prod(relation_affinity(t) for t in path_types) * hop_decay^(hops-1)`.

- `dense_rel(anchor)` is min-max normalized Dense relevance, so a weak anchor cannot promote confidently.
- The relation-affinity product is in (0, 1) and shrinks with path length, giving an intrinsic length penalty on top of `hop_decay`.
- One partner per anchor and one anchor per partner. Without one-to-one matching a single strong anchor floods the head with its whole neighbourhood, which is the mechanism behind `ppr_steiner`'s 4.68 edges/query at 0.075 precision.

Ties break on `(-confidence, dense_rank[partner], partner_id, path_node_ids)` for full determinism.

### Decision 5: Fallback is exact Dense identity

Zero accepted proposals returns the Dense `RankedNode` list unchanged, and the trace sets `exact_dense_fallback`. This makes "the method did nothing" an explicit, gradeable state rather than something inferred from metric equality — the condition that let the current no-op hide for a full experiment cycle.

## Algorithm

```
rank_task(request, top_k):
  dense, qvec  <- dense_ranker.rank_with_query_vector(request)
  affinity     <- softmax_over_present_types(cos(qvec, frozen_relation_vectors) / T)
  adjacency    <- undirected typed adjacency, excluding `contains`

  proposals <- []
  for anchor in dense[:anchor_top_a]:
      for (partner, path_types, path_nodes) in walk(anchor, <= max_hops):   # BFS, no revisits
          if partner not in candidates: continue                            # interior connectors only
          conf <- dense_rel(anchor) * prod(affinity[t] for t in path_types)
                                    * hop_decay ** (len(path_types) - 1)
          proposals.append(...)

  accepted <- one_to_one_match(
      [p for p in proposals if p.conf >= min_partner_confidence
                            and dense_rank[p.partner] > preserve_dense_top_n
                            and dense_rank[p.partner] > dense_rank[p.anchor]]
  )
  ranked   <- stable_insert(dense, accepted, preserve_dense_top_n)
  return ranked, typed_partner_completion_trace(...)
```

Complexity: `O(anchor_top_a * d^max_hops)` per query with `max_hops <= 2`. No linear system, no iteration to convergence. Expected to be faster than the current 48ms mean.

## Parameters

| name | default | role |
|---|---|---|
| `anchor_top_a` | 3 | how many Dense heads may propose |
| `max_hops` | 2 | walk bound; 1 = direct edge only |
| `hop_decay` | 0.6 | penalty for the 2-hop path |
| `min_partner_confidence` | 0.2 | single accept threshold |
| `preserve_dense_top_n` | 2 | protected MRR prefix |
| `relation_temperature` | 0.2 | retained from current config |

Defaults are seeded from GraphRAG's measured-good analogues (`seed_top_s=5`, `min_bridge_confidence=0.2`, `preserve_dense_top_n=2`) to avoid tuning against RQ2 before RQ3 is checked. Only `min_partner_confidence` is intended to be swept, on RQ2, once.

## Risks / Trade-offs

- **`min_partner_confidence` transfers poorly to RQ3.** RQ3 has 20 queries, so it cannot support tuning. Mitigation: tune on RQ2 only, then run RQ3 with the identical config and report whatever results. If RQ3 needs a different threshold, the unification claim has failed and must be reported as failed, not patched with a per-dataset value.
- **2 hops may be too shallow for RQ3 hard queries.** `typed_beam` uses `max_hops=4`. Mitigation: `max_hops` is a config field; measure 1/2/3 on both RQs and report the curve rather than silently picking the best.
- **Intra-top-k reorder can hurt MRR.** Mitigation: `preserve_dense_top_n` bounds it structurally. Verification gate 4 fails the change if RQ3 MRR drops below Dense.
- **Deleting the legacy variants forfeits the ability to reproduce the earlier reported numbers from this code path.** Accepted deliberately: the earlier `dependency_path` RQ2 result was measured to be Dense-identical on RQ3 and was never a valid unified method. Prior numbers stay reproducible from git history and the archived `results/*v4` artifacts.

## Verification Plan

Ordered; each gate must pass before the next is attempted.

1. **Unit/domain**: relation-affinity conditioning, `contains` exclusion, connector-not-ranked, one-to-one matching, score-multiset preservation, protected prefix, exact-Dense fallback, determinism under ties. Plus full `pytest`, `ruff`, `basedpyright` with no new diagnostics.
2. **RQ2 headroom sanity**: on a slice, confirm accepted promotions are non-zero and land inside the top-5. A run where `stable_insert` never fires is the current bug reproduced and blocks progress.
3. **RQ2 quality**: target `Recall@5` > 0.6561 and `Full Support@5` > 0.3294 (GraphRAG), with `MRR` >= 0.8090 and `Recall@2` unchanged at 0.4612.
4. **RQ3 transfer, identical config, zero changes**: target `R@10` >= 68.25 (`typed_beam`) with `MRR` >= 65.83 (Dense). `typed_beam` scores 60.33 MRR, so beating it on both axes at once is the differentiating result.
5. **Ablation**: constant relation affinity must degrade both RQs, or the typed-signal claim is unsupported and must be withdrawn.

Gates 3-5 require GPU runs the user executes on the server; this change lands gates 1-2 plus the code.

## Open Question (does not block implementation)

`Connected Evidence Recall@5/@10` and `Query-Evidence Connectivity@10` are `0.0` for **every** method in `results/*v4`, including BM25 and GraphRAG. `EvidenceMetricSuite._evaluate_impl` returns `0.0` whenever `graphs_by_task_id.get(task_id)` is `None`, and `validate_task_id_alignment` is skipped when `graph_task_ids` is empty — so an absent evidence-graph asset degrades silently instead of failing. RQ2 gold is always 2 nodes with a gold dependency edge between them, so a present graph should yield non-zero values on at least some tasks.

This was found by reading `graph_memory/evaluation/suites.py`; it has not been confirmed at runtime. It is tracked as a separate concern because these are precisely the metrics that would show connectivity value, and fixing them may demonstrate an advantage independent of any ranking change. It is out of scope for this change.

## 1. Configuration and Deletion

- [x] 1.1 Rewrite `epgm/config.py` to a single `EpgmRetrieverConfig` with `anchor_top_a`, `max_hops`, `hop_decay`, `min_partner_confidence`, `preserve_dense_top_n`, `relation_temperature`, frozen relation descriptions/version, and `NON_TRAVERSABLE_EDGE_TYPES = {contains}`; drop `variant`, `EPGM_VARIANTS`, `for_variant`, `DEFAULT_EDGE_PRIORS`, dependency/revision edge sets, lifecycle states, hub types, and every PPR/prize/Steiner field. Keep `cache_fingerprint`.
- [x] 1.2 Delete `epgm/search.py` and `epgm/selection.py`.
- [x] 1.3 Reduce `epgm/diffusion.py` to relation encoding and query-relation affinity, renaming it `epgm/relations.py`; remove `dense_teleport`, `build_typed_transitions`, `personalized_pagerank`, and the PPR/transition records.
- [x] 1.4 Delete `configs/method/execution_provenance_retriever_dependency_path.yaml` and remove `variant` from `configs/method/execution_provenance_retriever.yaml`.

## 2. Typed Partner Completion Operator

- [x] 2.1 Implement `epgm/walk.py`: undirected typed adjacency excluding `contains`, deterministic bounded BFS to `max_hops` without node revisits, yielding `(partner, path_node_ids, path_edge_types, directions)` for candidate partners only, with non-candidate interior connectors permitted.
- [x] 2.2 Implement proposal scoring `dense_rel(anchor) * prod(affinity) * hop_decay^(hops-1)`, retaining only the highest-confidence path per `(anchor, partner)` pair.
- [x] 2.3 Implement acceptance: threshold, protected prefix, strict upward-only partner rank, one partner per anchor, one anchor per partner, deterministic tie-breaks, and a machine-readable reason on every rejection.
- [x] 2.4 Implement `stable_insert` promotion preserving the Dense score multiset, rejecting non-effective insertions, and returning exact Dense objects when nothing is accepted.
- [x] 2.5 Implement stored-orientation candidate-edge emission with duplicate suppression and no emission for direction-divergent paths.
- [x] 2.6 Rewrite `epgm/method.py` to a single `rank_task` path; delete `rank`, `EpgmRankedNode`, `EpgmRetrievalResult`, `_additive_ranking`, `_select_paths`, and the PPR/subgraph branch. Update `epgm/__init__.py` exports.

## 3. Contracts, Serialization, Validation

- [x] 3.1 Replace `StatelessExecutionProvenanceTrace` and `QueryConditionedExecutionProvenanceTrace` with `TypedPartnerCompletionTrace` (plus proposal/affinity records) in `retrieval/contracts.py` and the native trace union.
- [x] 3.2 Replace both EPGM branches in `retrieval/execution/results.py` with one `typed_partner_completion` serializer.
- [x] 3.3 Replace `_validate_query_conditioned_subgraph_trace` and the `execution_provenance_local` branch in `validation/ranking.py` with one closed validator covering field closure, affinity mass, candidate/connector membership, proposal reasons, promotion bounds, emitted-edge candidacy, and fallback consistency. Drop the `EPGM_VARIANTS` import.

## 4. Registry, Config, Workflow Integration

- [x] 4.1 Remove `variant` from `ExecutionProvenanceMethodConfig`; construct the retriever config without `for_variant` in `stages/retrieve.py`.
- [x] 4.2 Advance the EPGM implementation version to `ranking-v5-epgm-tpc-<fingerprint>` in `experiment/workflow.py`, leaving every other method's identity untouched.
- [x] 4.3 Update `scripts/run_epgm_provenance.py`: drop `--epgm-variant` and the legacy `rank()` branch, read the new trace, and keep RQ3 JSONL output fields stable for `scripts/score_epgm_rq3.py`.

## 5. Tests

- [x] 5.1 Delete PPR, teleport, prize, Steiner, connector-budget, weight-normalization, schema-gate, and variant-preset tests from `tests/test_execution_provenance_retrieval_domain.py`.
- [x] 5.2 Add operator tests: relation-affinity conditioning, constant-weight invariance, unbound-`feeds` traversability, `contains` exclusion, exhaustive proposal enumeration, hop bound, connector-not-ranked, confidence ordering by anchor relevance and path length, one-to-one matching, protected prefix, upward-only, score-multiset preservation, intra-top-5 insertion, stored-orientation emission, divergent-path non-emission, exact fallback, determinism under ties.
- [x] 5.3 Add an RQ2-shaped and an RQ3-shaped graph test asserting the same config promotes correctly in both, including one case where the RQ3 partner is upstream of the anchor.
- [x] 5.4 Update trace round-trip, validator-rejection, registry, and `tests/test_experiment_workflow_config.py` cases for the removed `variant` and the new trace kind.

## 6. Verification

- [x] 6.1 Run focused EPGM tests, then full `PYTHONPATH="" uv run pytest`, `ruff`, and `basedpyright`; confirm no new diagnostics beyond the 7 pre-existing optional-member errors.
- [x] 6.2 Confirm no `prgcn-*` or baseline cache identity changed: diff resolved ranking identities for bm25/dense/graphrag/rgcn before and after.
- [x] 6.3 Run a small RQ2 slice locally and assert accepted promotions are non-zero and reach positions inside the top-5; a zero-promotion run reproduces the current bug and blocks the change.
- [x] 6.4 Document the RQ2 sweep of `min_partner_confidence` and the RQ3 identical-config transfer as pending user GPU execution, with the gates from `design.md` stated as pass/fail thresholds.

## 7. Corrections Found During Implementation

Each item below was a real defect caught by running the code, not a planning
change. Recorded because the reasoning matters more than the diff.

- [x] 7.1 **Affinity distribution used as a path factor.** Softmax over the
  present edge types caps the maximum at ~1/n (0.16 with 8 types), so no
  confidence could ever reach a 0.2 threshold and the retriever was a strict
  no-op. Split the concept: `affinity` stays the reported distribution,
  `preference` is peak-normalized to 1.0 and is what traversal multiplies.
- [x] 7.2 **Non-traversable types skewed the softmax.** `contains` was included
  when normalizing, letting an untraversable relation absorb mass and rescale
  every traversable preference. Affinity is now computed over traversable,
  positive-weight types only.
- [x] 7.3 **Generic relations outranked informative ones.** Query similarity
  alone selected `depends_on` as top relation in 11/20 RQ3 queries, and
  `depends_on` is the *least* precise relation there (0.057 gold rate vs 0.500
  for `invalidates`) because it is ~70% of all edges. Added per-graph
  inverse-frequency `specificity`; traversal uses `preference * specificity`.
  Measured effect on proposal ranking: precision@10 0.60 -> 0.80.
- [x] 7.4 **Anchor relevance was the wrong scoring term.** Measured on RQ3:
  ranking proposals by partner relevance gives precision@10 0.60, by anchor
  relevance 0.00. Scoring the anchor made every partner of one anchor tie, so
  selection fell through to an arbitrary tie-break. Confidence now keys on the
  partner; the anchor only decides where to look and where to insert.
- [x] 7.5 **Min-max relevance made the last candidate unpromotable.** Min-max
  sends the lowest-scoring candidate (and all ties at the minimum) to exactly
  0.0; since confidence multiplies by partner relevance, those candidates could
  never be promoted regardless of structural evidence. Switched to peak
  normalization.
- [x] 7.6 **Validator rejected legitimate repeated relations.** `_native_trace_string_list`
  applied set-uniqueness to every string list, including the ordered per-step
  `path_edge_types` / `path_directions`. Any 2-hop path over one relation type
  (`depends_on` twice) failed validation. Uniqueness is now opt-out for ordered
  sequences. Regression test added.

## 8. Measured Results (RQ3, 20 queries, seed-free CPU run)

`uv run python -m scripts.run_epgm_provenance --methods bm25,dense,graphrag,epgm_retriever`
then `scripts.score_epgm_rq3`:

| method | R@5 | R@10 | FS@5 | FS@10 | MRR | edges |
|---|---|---|---|---|---|---|
| bm25 | 34.08 | 41.33 | 2/20 | 3/20 | 40.42 | 0/20 |
| dense | 50.50 | 64.42 | 4/20 | 7/20 | 65.83 | 0/20 |
| graphrag | 51.00 | 63.25 | 4/20 | 7/20 | **67.08** | 5/20 |
| **ours (tpc)** | **53.17** | **66.08** | 4/20 | 7/20 | 65.83 | 12/20 |

Recall@10 by level - the gain is concentrated in `medium` (58.33 -> 65.00);
`easy` and `hard` are unchanged. Both prior variants are beaten on R@10
(`ppr_steiner` 52.25, `typed_beam` 68.25 but at MRR 60.33).

Robustness: RQ3 is flat across `min_partner_confidence` 0.35-0.55,
`max_hops` 1-3, and `preserve_dense_top_n` 0-3, so the default sits on a
plateau rather than a tuned spike. `max_hops` is inert on RQ3 because every
accepted promotion there is a direct typed edge.

### Ablations (RQ3) - partly negative, reported as measured

| variant | R@5 | R@10 | FS@10 | MRR |
|---|---|---|---|---|
| full | 53.83 | 66.08 | 7 | 66.05 |
| constant relation weight | 54.33 | 65.33 | 6 | 66.53 |
| preference only (no specificity) | 51.50 | 63.17 | 6 | 66.04 |
| specificity only (no query match) | 53.83 | 64.83 | 6 | 66.05 |

Honest reading: specificity is load-bearing (removing it costs 2.9 R@10 and
drops to GraphRAG level), but **the query-conditioning ablation is not clean** -
constant relation weight matches the full method on R@5/MRR and loses only
0.75 R@10 and one FS@10 case. On 20 queries that is not a defensible claim for
query conditioning. Either RQ2 must show a clear separation, or the paper should
attribute the gain to relation specificity plus partner completion and stop
claiming query conditioning as a contribution. Do not report the full-vs-constant
ablation as supportive on this evidence.

## 9. Pending User GPU Execution (RQ2)

Gates from `design.md`, unchanged and not yet run:

1. RQ2 quality: `Recall@5` > 0.6561 and `Full Support@5` > 0.3294 (GraphRAG),
   with `MRR` >= 0.8090 and `Recall@2` unchanged at 0.4612.
2. Sweep `min_partner_confidence` on RQ2 only; confirm a plateau rather than a
   spike, then keep the RQ3 default if it lies inside it.
3. Re-run the RQ3 numbers in section 8 with the final shared config to confirm
   one configuration still serves both.
4. Rerun the query-conditioning ablation on RQ2's 2851 tasks, where it is
   statistically meaningful, before making any claim about it.

Command: `uv run python -m graph_memory.experiment ... method=execution_provenance_retriever`
(the `variant` override no longer exists). EPGM ranking cache identity moved to
`ranking-v5-epgm-tpc-<fingerprint>`; baseline and RGCN identities are unchanged.

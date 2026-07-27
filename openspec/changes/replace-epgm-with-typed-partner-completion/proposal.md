## Why

Seed-13 RQ2 per-task analysis (`results/*-twp-sd13-v4`, 2851 tasks) shows the current `ppr_steiner` default is a near-identity function, not a weak method: it changes `Recall@10` on **9/2851** tasks and `Recall@5` on **1/2851**, for aggregate deltas of `-0.0002`. GraphRAG changes 126 and 253 tasks respectively for `+0.0196` and `+0.0405`. On RQ3 (`runs/epgm_rq3/task2`) the same code is actively harmful: `R@10` drops to 52.25 against Dense 64.42 and `typed_beam` 68.25. The historical `dependency_path` variant is byte-identical to Dense on RQe3, so the claimed RQ2/RQ3 unification never held — each legacy variant no-ops on one of the two datasets.

Three root causes were confirmed by reading the code and by executing a minimal reproduction of `select_budgeted_subgraph`:

1. **Promotions can never enter the top-5.** `_rank_connected_subgraph` rebuilds the head as `[*top_ids, *tail_ids]` where `top_ids` follows Dense order. A promoted candidate has Dense rank > `top_k` by construction, so it always lands in the last in-budget slot. Structure can therefore change top-k *membership* but never intra-top-k *order*. All available RQ2 headroom is intra-top-10: 781 tasks (27.4%) already hold every gold node in the Dense top-10 but not in the top-5, and a top-10 oracle reorder lifts `Recall@5` 0.6156 -> 0.7601 and `Full Support@5` 0.2473 -> 0.5212.
2. **The greedy loop spends its budget on incumbents.** Each step adds every not-yet-selected candidate along the connector path. A single long first path consumes the whole `evidence_budget`, terminating the loop before any graph-central out-of-budget candidate is ever proposed. A 20-node reproduction shows a PPR-dominant node at Dense rank 15 is only reachable when a direct arc from the root exists.
3. **One constant set cannot serve two graph densities.** `prize = 0.55*dense + 0.45*ppr - 0.2` plus `selection_edge_cost_weight`/`connector_hop_cost`/`selection_min_gain` are tuned against RQ2 pool statistics; on the smaller, sharper RQ3 graphs the identical constants over-trigger. This is the direct cause of the simultaneous RQ2 no-op and RQ3 regression.

Edge quality confirms the objective is mis-specified rather than under-tuned: `ppr_steiner` emits 4.68 edges/query at `Edge Precision@10` 0.075 (`F1` 0.124) while GraphRAG emits 0.29 at precision 0.342 (`F1` 0.156) — 16x the edges for lower F1.

Both research questions share one evidence shape. RQ2 gold is always 2 sentences joined by exactly 1 gold dependency edge; RQ3 gold is a conclusion plus the source evidence supporting it. The single operation both need is *typed partner completion*: from a Dense-confident anchor, walk query-relevant typed edges a bounded number of hops and lift the anchor's evidence partner to just after it. This replaces PPR, Steiner selection, prize/displacement accounting, and hand-written schema gates with one auditable operator that keeps the query-conditioned relation affinity — the one component of the current architecture that is genuinely dataset-agnostic and worth retaining.

## What Changes

- Replace the EPGM retriever with a single `typed_partner_completion` algorithm; delete the `ppr_steiner`, `typed_beam`, and `dependency_path` variants and the `variant` config axis entirely.
- Delete `diffusion.py` PPR machinery (`dense_teleport`, `build_typed_transitions`, `personalized_pagerank`), all of `selection.py`, and all of `search.py`; retain and relocate only query/relation affinity encoding.
- Propose partners by bounded undirected typed walk (`max_hops` <= 2) from the Dense top-`anchor_top_a` anchors, allowing non-candidate nodes as interior connectors that never consume ranking slots.
- Score each proposal as `dense_relevance(anchor) * prod(relation_affinity(edge_type)) * hop_decay^(hops-1)`, where `relation_affinity` is the existing frozen query-to-relation-description softmax. No recorded `edge.weight` magnitude, no field-binding gate, no hand-authored per-type priors, no lifecycle gate.
- Accept proposals above one confidence threshold with at most one partner per anchor and one anchor per partner, protecting the Dense top-`preserve_dense_top_n` prefix.
- Apply promotions by stable insertion directly after the anchor, so structure **may** reorder within the top-k. This is the mechanism that converts the measured intra-top-10 headroom into `Recall@5`/`Full Support@5` gains.
- Reduce the tunable surface to `anchor_top_a`, `max_hops`, `min_partner_confidence`, `hop_decay`, and `preserve_dense_top_n`; all remain cache-identity fields.
- Replace the `execution_provenance_subgraph` native trace with a `typed_partner_completion` trace carrying relation affinities, every proposal with accept/reject reason, connector node ids, and fallback state; remove the `execution_provenance_local` trace kind.
- Delete `configs/method/execution_provenance_retriever_dependency_path.yaml`.

## Capabilities

### New Capabilities
- `typed-partner-completion-retrieval`: One non-trained EPGM operator that completes an evidence partner set from Dense anchors using query-conditioned typed relations, shared unchanged across synthetic dependency graphs (RQ2) and real agent traces (RQ3).

### Modified Capabilities
- `execution-provenance-retrieval`: Replaces variant-selected diffusion/subgraph retrieval with one partner-completion operator and removes the `variant` axis from the public method config.
- `method-native-retrieval-traces`: Replaces both EPGM trace kinds with one closed partner-completion trace.

### Removed Capabilities
- `query-conditioned-provenance-subgraph-retrieval`: PPR diffusion, budgeted Steiner extraction, prize/displacement accounting, and the typed-beam/dependency-path diagnostics are removed rather than retained as ablations, because both legacy variants are measurably identity functions on one of the two target datasets.

## Impact

- Primary code: `graph_memory/retrieval/methods/epgm/` (net reduction from 2257 lines), `graph_memory/retrieval/contracts.py`, `graph_memory/retrieval/execution/results.py`, `graph_memory/validation/ranking.py`, `graph_memory/experiment/config.py`, `graph_memory/stages/retrieve.py`, `graph_memory/registry/retrieval*.py`, `scripts/run_epgm_provenance.py`.
- Config: `configs/method/execution_provenance_retriever.yaml` loses `variant`; the dependency-path config file is deleted.
- Cache identity: the `ranking-v4-epgm-<fingerprint>` prefix advances to `ranking-v5-epgm-tpc-<fingerprint>`, invalidating only `execution_provenance_retriever` ranking artifacts. BM25, Dense, GraphRAG, evidence-RGCN, and provenance-RGCN identities are untouched, so existing `prgcn-*` and baseline caches stay valid.
- Tests: EPGM domain tests are rewritten against the new operator; PPR/Steiner/prize/gate tests are deleted. Registry, serializer, validator, workflow-config, and RQ3 scoring tests are updated.
- No trainable parameters, no dataset-specific switches, no label access, no graph regeneration, no data migration.

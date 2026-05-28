## Context

Phase 1 currently has a stable public retrieval contract: `bm25`, `dense`, `bm25_graph_rerank`, and `dense_graph_rerank` all produce ranked results in the shared schema. A previous refactor introduced `ScorePipelineMethod` and node-score components so flat and graph-rerank methods can share score-combination behavior.

The remaining architecture issue is ownership. `graph_memory/retrieval.py` now contains both retrieval orchestration and graph-rerank score-component classes, while `graph_memory/rerank.py` still owns candidate expansion, graph score helpers, and compatibility functions such as `graph_rerank(...)`. This split makes it unclear which file is authoritative for graph-rerank scoring.

The config naming adds a second ambiguity. `GraphRerankConfig.type_weights` is serialized next to `lambda_init`, `lambda_query`, `lambda_neighbor`, and `lambda_bridge`, but it does not mean the same thing. Lambdas weight final score components. Type weights calibrate graph edge types inside neighbor-style graph scoring. Keeping the name `type_weights` makes tuned configs look like they contain duplicate weight layers.

## Goals / Non-Goals

**Goals:**

- Make `graph_memory/rerank.py` the authoritative home for graph-rerank scoring abstractions and helpers.
- Keep `graph_memory/retrieval.py` focused on public method orchestration, seed retriever selection, latency measurement, and result assembly.
- Rename `type_weights` to `neighbor_type_weights` in graph-rerank config objects and newly written config artifacts.
- Make the semantics explicit: `lambda_*` fields weight final score components; `neighbor_type_weights` calibrates graph edge types inside neighbor propagation and bridge-neighbor scoring.
- Ensure query-overlap scoring remains controlled only by `lambda_query` and does not consume `neighbor_type_weights`.
- Preserve existing ranking behavior after config migration.
- Reject deprecated `type_weights` input so old configs must be converted before reuse.

**Non-Goals:**

- Do not implement a new graph-rerank formula.
- Do not tune `neighbor_type_weights` in this change.
- Do not change graph construction, graph edge schema, evaluation metrics, public method names, or ranked-result schema.
- Do not implement path scoring or make `lambda_path` active.
- Do not delete historical run artifacts.

## Decisions

### Decision 1: Move graph-rerank score pipeline pieces into `rerank.py`

`rerank.py` should own graph-rerank scoring because it already owns candidate expansion, query-overlap score extraction, neighbor propagation, bridge scores, induced subgraph extraction, and compatibility helpers. The graph-aware score components should sit next to those helpers so the graph-rerank formula has one canonical module.

The intended boundary is:

```text
retrieval.py
  select seed retriever
  build public method runner
  measure latency
  assemble RankedResult

rerank.py
  normalize graph-rerank components
  expand graph candidates
  compute graph component scores
  combine graph-rerank components
  expose graph-rerank runner/helpers
```

Alternative considered: keep `ScorePipelineMethod` in `retrieval.py` because flat methods also use it. That keeps all retrieval methods in one file, but it leaves graph-specific abstractions far from the graph helpers they depend on. Since flat methods only need trivial one-component ranking, the graph-rerank-specific pipeline should move and retrieval orchestration can call it.

### Decision 2: Use `neighbor_type_weights` for edge-type calibration

Rename `type_weights` to `neighbor_type_weights` in `GraphRerankConfig` and in generated config JSON. This makes the scope visible at the call site: the field belongs to neighbor-style graph propagation, not to final score-component weighting.

The renamed field should include only edge types that can participate in neighbor-style memory-to-memory scoring. `query_overlap` should not be required because query-overlap is a separate `q -> memory` component controlled by `lambda_query`.

Alternative considered: keep the field name and document it better. That avoids migration work but leaves tuned config JSON misleading, which is the current failure mode.

### Decision 3: Keep query overlap independent from neighbor type weights

Query-overlap component score should be:

```text
lambda_query * normalized(S_query)
```

It should not become:

```text
lambda_query * neighbor_type_weights["query_overlap"] * normalized(S_query)
```

The second form creates duplicate controls for the same component and makes ablations hard to reason about. If query-overlap strength needs tuning, tune `lambda_query`.

### Decision 4: Reject old configs, write new configs

Config loading should accept the canonical shape:

```json
{"neighbor_type_weights": {"sequential": 0.3, "entity_overlap": 0.7, "bridge": 1.0}}
```

and reject deprecated input:

```json
{"type_weights": {"query_overlap": 0.8, "sequential": 0.3, "entity_overlap": 0.7, "bridge": 1.0}}
```

When both are present, loading should still fail so there is one visible config spelling. Newly written selected configs and candidate rows should emit `neighbor_type_weights` only. Historical configs with `type_weights` must be converted before rerunning graph-rerank commands.

### Decision 5: Keep behavior parity covered by tests

The refactor should prove that moving code does not change ranking behavior. Focused tests should compare graph-rerank output before and after the boundary change using tiny graphs and fixed initial scores. Config migration tests should prove that deprecated `type_weights` input fails clearly and canonical `neighbor_type_weights` drives memory-to-memory graph edge calibration.

## Risks / Trade-offs

- [Risk] Moving score components can introduce circular imports between `retrieval.py`, `rerank.py`, and `types.py`. -> Mitigation: keep shared dataclasses and type aliases in `types.py`; keep rerank helpers independent from BM25/dense retriever implementations.
- [Risk] Config rename can break historical selected config files. -> Mitigation: fail clearly on deprecated `type_weights` and document that old configs must be converted before reuse.
- [Risk] Removing `query_overlap` from required type weights changes validation expectations. -> Mitigation: update validation and tests together so query-overlap remains validated through `lambda_query` and graph edge validation, not neighbor type weights.
- [Risk] The term `neighbor_type_weights` may sound like it excludes bridge scoring. -> Mitigation: document bridge score as bridge-neighbor graph scoring and include `bridge` in the default neighbor type weights.
- [Risk] Rejecting compatibility input makes old runs less convenient to reproduce. -> Mitigation: keep historical artifacts immutable and require an explicit conversion step before rerunning them.

## Migration Plan

1. Add tests for `neighbor_type_weights` canonical config, deprecated `type_weights` rejection, and query-overlap independence.
2. Move graph-rerank score component classes and graph-rerank combination helpers from `retrieval.py` into `rerank.py`.
3. Replace `GraphRerankConfig.type_weights` with `neighbor_type_weights`, and reject old dict records that still contain `type_weights`.
4. Update tuning grid parsing so search-space JSON reads `neighbor_type_weights` and selected output writes `neighbor_type_weights`.
5. Update validation and docs to remove `query_overlap` from neighbor type weight requirements.
6. Run focused retrieval/rerank tests, config validation tests, CLI smoke tests that read the graph-rerank grid, and OpenSpec validation.

Rollback is straightforward before generated artifacts are published: revert the source changes and keep old `type_weights` configs. After publishing new config artifacts, rollback requires converting generated `neighbor_type_weights` configs back to `type_weights`.

## Open Questions

- How long should deprecated `type_weights` input remain accepted? Answer for implementation: do not accept it; require conversion before reruns.
- Should historical `runs/**` artifacts be mass-renamed? Recommended answer for implementation: no; treat them as immutable run outputs and convert only copied configs that will be reused.

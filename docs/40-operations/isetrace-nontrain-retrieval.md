# ISETrace non-training retrieval pilot

This workflow evaluates reviewed or provisional natural queries over two query-independent retrieval views of the same canonical ISETrace trajectory. It does not train a model.

## Inputs

```text
data/isetrace/query-authoring/test-span-pilot-v5.jsonl
data/isetrace/splits/v1/test.trajectories.jsonl
ISETrace revision e40e04d41c04e4eb4bae181ebdd41b61c688081b
```

`configs/dataset/isetrace.yaml` is deliberately pilot-only:

```yaml
review_policy: allow_unreviewed
label_policy: intent_aware
```

The checked-in pilot query file predates the exact-span contract and is intentionally rejected. Generate a v5 span-labeled authoring artifact before running this workflow. Provisional LLM records remain `unreviewed`; the configuration may permit engineering runs but does not make their labels formal benchmark gold. Formal runs must use a manually reviewed query file and `review_policy=accepted_only`.

Under `intent_aware`, `complete_chain` and `contributing_sources` use complete support spans. Directional intents use answer spans. Per-task output retains `query_intent`, `motif_type`, and review status so complete-support subsets can be reported separately. Legacy records containing only answer/support output IDs have no fallback and fail validation.

## Retrieval views and graph boundary

Every method sees the same canonical trajectory, source revision, query, and source coordinate system. Candidate IDs are method-local and are not compared across methods.

- BM25, Dense, and GraphRAG rank fixed token-window chunks of the complete rendered trajectory, including messages, complete raw tool-call arguments, and complete tool outputs. The configured 512-token encoder window reserves eight tokens for model prefix/special tokens, leaving 504 source tokens per chunk with 64-token overlap.
- GraphRAG builds its private entity graph from those flat chunks. It never receives native provenance edges.
- `provenance_path` ranks argument and output content chunks from the execution-provenance graph. ToolCall/ToolOutput roots, artifacts, and typed edges participate in bounded traversal but are not returned as free evidence.

The physical `ProvenanceGraph` is built once per trajectory before query projection. It contains no query or gold fields. ToolCall and ToolOutput roots contain only lightweight identity text; full arguments and outputs live exclusively in source-backed content chunks linked by `has_argument`, `has_content`, and `next_chunk` edges.

All ranked records carry exact `(event_id, json_pointer, char_start, char_end)` source spans. Evaluation unions those coordinates rather than comparing candidate IDs or searching for gold text inside a node.

## Methods

- `bm25`: lexical retrieval over flat trajectory chunks.
- `dense`: frozen E5 retrieval over the same flat trajectory chunks.
- `graphrag`: private text-derived entity graph over flat trajectory chunks; native execution-provenance edges are unavailable to the method.
- `provenance_path`: frozen-Dense seeds over provenance content units plus bounded bidirectional traversal through the native execution graph. Temporal `precedes` edges are excluded; native call/content, return, data-flow, artifact, and chunk-order edges remain available.

## Commands

Smoke runs:

```powershell
uv run python experiment/run.py name=isetrace_bm25_smoke dataset=isetrace profile=smoke method=bm25 device=cpu
uv run python experiment/run.py name=isetrace_dense_smoke dataset=isetrace profile=smoke method=dense device=cpu
uv run python experiment/run.py name=isetrace_graphrag_smoke dataset=isetrace profile=smoke method=graphrag device=cpu
uv run python experiment/run.py name=isetrace_path_smoke dataset=isetrace profile=smoke method=provenance_path device=cpu
```

Pilot multirun:

```powershell
uv run python experiment/run.py -m `
  name=isetrace_nontrain_pilot dataset=isetrace profile=full device=cuda:0 `
  method=bm25,dense,graphrag,provenance_path
```

The first full run should keep method defaults frozen. Do not tune GraphRAG or provenance-path parameters against this natural-query test pilot.

## Interpretation

Report top-k span Coverage/Recall, Span F1, Full Support, Evidence Density, and MRR. Evidence Density charges every returned source-backed span, so duplicate overlap lowers density even though coverage is unioned. Also report Coverage, Span F1, Full Support, and Evidence Density under the fixed 2048-token context budget; top-k alone is not comparable across different retrieval-unit granularities.

The aggregate mixes answer-event and complete-support tasks and must not be reported alone as a complete-support result. Report at least:

- all-query answer/evidence recall;
- `complete_chain` + `contributing_sources` Full Support and path metrics;
- per-intent metrics;
- GraphRAG exact-Dense fallback/bridge activation;
- provenance-path exact-Dense fallback, accepted paths, and emitted dependency edges.

`Query-Evidence Connectivity@10` is `N/A`: ISETrace graphs intentionally contain no query-conditioned topology.

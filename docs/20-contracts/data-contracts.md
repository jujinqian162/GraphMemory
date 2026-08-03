# Data contracts

## Evidence workflow artifacts

| Artifact | Producer | Consumer |
|---|---|---|
| prepared task/label payloads | `stages.prepare` | graph, pair, rank, train, eval |
| `EvidenceGraph` payload | `stages.graphs` | evidence R-GCN + graph-dependent eval |
| training pairs | `stages.pairs` | Dense-FT or R-GCN training |
| ranked predictions | `stages.retrieve` | evaluation |

Reusable payloads are immutable under `data/processed/`. Prefect stores small typed references (kind, URI, digest, origin). `runs/<name>/assets/manifest.yaml` is delivery-only, never a computation input.

EvidenceGraph construction uses input-visible question/candidate fields only. Labels never enter retrieval graph construction.

## Execution provenance

The synthetic 2Wiki provenance adapter, its transform, and the legacy shared execution-provenance contracts have been removed. Standard `twowiki` remains an evidence dataset and is unchanged.

ISETrace has one prepared benchmark path shared by non-training methods and the trainable provenance R-GCN:

```text
ISETrace JSONL
  -> CanonicalTrajectory
  -> query-independent ProvenanceGraph
  -> deterministic task text with A/E handles
  -> minimal LLM authoring record: id + text + query + gold quotes
  -> exact SourceSpan compilation during benchmark preparation
  -> frozen trajectory-grouped split + explicit natural/template counts
  -> ISETraceRankingRecord + span-only ISETraceLabelRecord
  -> BM25 / Dense / GraphRAG / provenance_path / provenance_rgcn
```

`graph_memory.datasets.isetrace` owns strict raw records, streaming parsing, deterministic adaptation, and compact ingestion counters. `graph_memory.trajectories` owns dataset-neutral ordered message/tool events. Source `success` is preserved only as `source_reported_success`; it is not an authoritative failure/retry label.

`ProvenanceGraph` is one immutable graph per trajectory. It contains no query, answer, motif, support label, or edge weight. The v1 core builder emits tool-call, tool-output, and artifact nodes with native/deterministic returns, temporal, exact-feed, read, and write relations. Kinds and relations are namespaced, and nodes retain canonical source spans so future annotators can add `semantic.claim` or `semantic.decision` layers without mutating source events or pretending those fields were native.

The durable natural-query JSONL has four fields: `id`, handle-delimited task `text`, natural-language `query`, and `gold` entries containing `{source, quote}`. Preparation first resolves valid records and assigns trajectory groups using registered v7 ownership weights plus `split_seed`; model seeds and supervision counts do not change ownership. `queries.splits.<split>.natural/template` then requests exact task counts, including natural-zero template-only train/dev. Natural gold remains exact `SourceSpan` data compiled from quotes. Deterministic `MotifSpec` records render template supervision whose labels are only focused ToolOutput content candidates. Nonfocused motif participants stay eligible negatives. Template supervision and `query_origin` are sidecar payloads, never request, candidate, graph, or model-feature fields. Test requires natural-only counts.

The LLM authoring utility assigns each task a deterministic writing style and emits provisional v7 records for review; see [`../40-operations/isetrace-query-authoring.md`](../40-operations/isetrace-query-authoring.md). Retired answer/support/intent-aware policies and legacy label-conditioned query envelopes are not restored.

Preparation content-addresses the natural-query JSONL and pinned trajectory source, whose revision is inferred from dataset registration and checked against authoring metadata when available. It materializes flat chunks, provenance argument/output chunks, immutable graphs, exact labels, origins, template sidecars, and deterministic counts. BM25, Dense, and GraphRAG receive flat text requests; `provenance_path` receives the physical graph without training; `provenance_rgcn` receives physical graphs and provenance content candidates through the shared trainable lifecycle. Every test method is evaluated against the derived natural-only exact spans.

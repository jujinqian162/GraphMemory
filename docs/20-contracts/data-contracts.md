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

ISETrace has a domain-library path plus a test-only non-training experiment adapter:

```text
ISETrace JSONL
  -> CanonicalTrajectory
  -> query-independent ProvenanceGraph
  -> deterministic task text with A/E handles
  -> minimal LLM authoring record: id + text + query + gold quotes
  -> exact SourceSpan compilation during benchmark preparation
  -> ISETraceRankingRecord + span-only ISETraceLabelRecord
  -> BM25 / Dense / GraphRAG / provenance_path
```

`graph_memory.datasets.isetrace` owns strict raw records, streaming parsing, deterministic adaptation, and compact ingestion counters. `graph_memory.trajectories` owns dataset-neutral ordered message/tool events. Source `success` is preserved only as `source_reported_success`; it is not an authoritative failure/retry label.

`ProvenanceGraph` is one immutable graph per trajectory. It contains no query, answer, motif, support label, or edge weight. The v1 core builder emits tool-call, tool-output, and artifact nodes with native/deterministic returns, temporal, exact-feed, read, and write relations. Kinds and relations are namespaced, and nodes retain canonical source spans so future annotators can add `semantic.claim` or `semantic.decision` layers without mutating source events or pretending those fields were native.

Motifs are used only by the offline generator to select coherent source events; they do not define v7 benchmark labels. One authoring task may produce two independent query/exact-gold groups whose evidence sets may differ. The durable LLM authoring JSONL has four fields: `id`, handle-delimited task `text`, natural-language `query`, and `gold` entries containing `{source, quote}`. Deterministic benchmark preparation matches the task text to the pinned trajectory and converts each quote directly into a canonical `SourceSpan`. V7 has no answer/support label variants, output-ID gold sets, dependency-gold reconstruction, or compiled motif sidecar. Direct, linked, and multi-fact authoring strata may be retained in an operational query-ID mapping for grouped reporting, but they are neither gold nor retrieval features.

The LLM authoring utility assigns each task a deterministic writing style and emits provisional v7 records for review; see [`../40-operations/isetrace-query-authoring.md`](../40-operations/isetrace-query-authoring.md). The retired template catalog and legacy query/label/generation envelopes are not part of this path.

The non-training adapter content-addresses the v7 query JSONL and pinned trajectory JSONL. It materializes flat trajectory chunks and provenance argument/output chunks, while every method is evaluated against the same exact `gold_evidence_spans` compiled only from `{source, quote}`. Provenance graphs remain query-independent retrieval inputs for `provenance_path`; graph dependencies never create, split, or expand v7 gold. BM25, Dense, and GraphRAG receive text requests, and `provenance_path` additionally receives the physical graph.

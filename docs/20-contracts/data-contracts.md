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

ISETrace now has a domain-library path, deliberately outside the experiment workflow:

```text
ISETrace JSONL
  -> CanonicalTrajectory
  -> ProvenanceGraph
  -> MotifSpec
  -> SyntheticQueryRecord + SyntheticQueryLabel
```

`graph_memory.datasets.isetrace` owns strict raw records, streaming parsing, deterministic adaptation, and compact ingestion counters. `graph_memory.trajectories` owns dataset-neutral ordered message/tool events. Source `success` is preserved only as `source_reported_success`; it is not an authoritative failure/retry label.

`ProvenanceGraph` is one immutable graph per trajectory. It contains no query, answer, motif, support label, or edge weight. The v1 core builder emits tool-call, tool-output, and artifact nodes with native/deterministic returns, temporal, exact-feed, read, and write relations. Kinds and relations are namespaced, and nodes retain canonical source spans so future annotators can add `semantic.claim` or `semantic.decision` layers without mutating source events or pretending those fields were native.

Motif and query artifacts remain separate from the graph. A motif can expose multiple query intents with intent-specific answer/support IDs; the verbalizer receives only explicitly safe slots. The versioned v1 catalog contains six templates across six writing styles for every supported motif/query-intent pair. ISETrace remains absent from Hydra dataset choices and Prefect stages until a later change defines fixed split artifacts, retrieval requests, model inputs, evaluation, and natural-query validation.

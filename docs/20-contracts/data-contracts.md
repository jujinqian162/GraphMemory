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

The synthetic 2Wiki provenance adapter, its transform, and the legacy shared execution-provenance contracts have been removed. In particular, the codebase no longer defines query `Task` nodes, `Answer`/`Claim`/`Decision` nodes, field-bound semantic relation keys, or weighted provenance edges.

ISETrace currently has revision-pinned raw download provisioning only. Canonical trajectory, graph, query, motif, and label contracts will be introduced after data auditing. Standard `twowiki` remains an evidence dataset and is unchanged.

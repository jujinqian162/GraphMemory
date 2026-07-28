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

`ExecutionProvenanceGraph` is source-native. Core node types: Task, Agent, ToolCall, ToolOutput, Answer. Core edges: `invokes`, `returns`, `feeds` (requires field binding), `grounds`; `precedes` is chronology only.

The only concrete dataset is synthetic `twowiki_provenance`. Each raw record splits into:

- **ranking** — question, ToolOutput candidates, typed graph (no gold flags)
- **label** — answer, ordered gold output IDs, contracted dependency edge

Transform runs in-flow (`stages.transform` / `transform_twowiki_task`) before prepare. Version tag `v{schema_version}-{digest}` encodes schema, transform params, and encoder identity. Standard `twowiki` remains an evidence dataset and is unchanged.

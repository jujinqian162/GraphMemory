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

The synthetic 2Wiki provenance adapter and its in-flow transform have been removed. ISETrace is the next source dataset; raw download provisioning exists, while its canonical trajectory, graph, query, and label contracts are introduced in subsequent migration stages. Standard `twowiki` remains an evidence dataset and is unchanged.

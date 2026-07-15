# Project overview

The project studies whether explicit graph structure improves recovery of complete evidence or execution paths beyond flat semantic retrieval. The governing implementation plan is [`execution-provenance-retrieval-domain-plan.md`](../10-plans/execution-provenance-retrieval-domain-plan.md).

## Supported domains

| Domain | Inputs | Methods |
| --- | --- | --- |
| Evidence retrieval | text candidates; optional `EvidenceGraph` for R-GCN | BM25, Dense, Dense-FT, GraphRAG, Dense R-GCN, Dense-FT R-GCN |
| Execution provenance | flat text requests or native `ExecutionProvenanceRankingRequest` | BM25, Dense, GraphRAG, Execution-Provenance Retriever |

`EvidenceGraph` contains question/evidence nodes and evidence relations. `ExecutionProvenanceGraph` contains Task, Agent, ToolCall, ToolOutput, and Answer nodes plus typed execution/dataflow edges. GraphRAG owns a third, private entity graph and does not consume either public graph artifact.

HotpotQA, 2Wiki, and MuSiQue remain evidence-retrieval datasets. The TRAJECT-Bench adapter evaluates offline tool retrieval against the benchmark's public domain catalogs. It exposes tools as text candidates and preserves catalog-declared connections for optional graph consumers, while the gold calls, arguments, outputs, final answer, and trajectory order remain label-only. It therefore does not claim to reproduce TRAJECT-Bench's end-to-end agent execution metrics or to provide a native `ExecutionProvenanceGraph`.

## Research boundary

- Flat methods establish lexical and semantic baselines.
- GraphRAG tests method-owned entity propagation without a dataset graph artifact.
- The two R-GCN methods learn independent evidence-node scores from `EvidenceGraph` inputs.
- The provenance retriever combines semantic seeds with typed path expansion, field binding, grounding, and revision invalidation.
- Retrieval outputs preserve complete rankings. Native traces contain only edges and paths actually used by the method.

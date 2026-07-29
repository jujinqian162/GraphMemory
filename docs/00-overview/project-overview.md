# Project overview

Graph structure should help recover complete evidence sets or execution paths beyond flat semantic retrieval.

## Domains

| Domain | Datasets | Methods |
|---|---|---|
| Evidence retrieval | HotpotQA, 2Wiki, MuSiQue | BM25, Dense, Dense-FT, GraphRAG, Dense R-GCN, Dense-FT R-GCN |
| Execution provenance | ISETrace (adapter in progress) | EPGM, Provenance R-GCN |

Domains do not project into each other.

| Graph | Owner | Consumers |
|---|---|---|
| `EvidenceGraph` | dataset stage | evidence R-GCN only |
| GraphRAG entity graph | GraphRAG method (private) | GraphRAG only |
| `ExecutionProvenanceGraph` | request / dataset | EPGM and Provenance R-GCN |

## Research boundary

- Flat methods are lexical/semantic baselines.
- GraphRAG tests method-owned entity propagation without a dataset graph artifact.
- Evidence R-GCN learns independent node scores on `EvidenceGraph`.
- EPGM (`execution_provenance_retriever`) is non-trained; default variant `ppr_steiner`.
- Provenance R-GCN ranks ToolOutput candidates with relation-aware message passing and separate edge scoring; no beam decoder.
- Every method returns a full ranking. Native traces contain only edges/paths the method actually used.

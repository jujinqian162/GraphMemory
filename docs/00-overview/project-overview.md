# Project overview

Graph structure should help recover complete evidence sets or execution paths beyond flat semantic retrieval.

## Domains

| Domain | Datasets | Methods |
|---|---|---|
| Evidence retrieval | HotpotQA, 2Wiki, MuSiQue | BM25, Dense, Dense-FT, GraphRAG, Dense R-GCN, Dense-FT R-GCN |
| Execution provenance | ISETrace (raw download only) | redesign pending |

The evidence workflow remains runnable. ISETrace currently has revision-pinned raw download provisioning only; it is intentionally absent from the experiment dataset/config registry until its trajectory-native contracts are defined.

| Graph | Owner | Consumers |
|---|---|---|
| `EvidenceGraph` | dataset stage | evidence R-GCN only |
| GraphRAG entity graph | GraphRAG method (private) | GraphRAG only |

## Research boundary

- Flat methods are lexical/semantic baselines.
- GraphRAG tests method-owned entity propagation without a dataset graph artifact.
- Evidence R-GCN learns independent node scores on `EvidenceGraph`.
- The legacy synthetic-provenance EPGM and Provenance R-GCN implementations were removed rather than carried into ISETrace.
- New provenance contracts must be derived from audited execution trajectories and must not reintroduce query nodes, answer nodes, semantic edge weights, or dataset-specific relation vocabularies.

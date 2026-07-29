# Project overview

Graph structure should help recover complete evidence sets or execution paths beyond flat semantic retrieval.

## Domains

| Domain | Datasets | Methods |
|---|---|---|
| Evidence retrieval | HotpotQA, 2Wiki, MuSiQue | BM25, Dense, Dense-FT, GraphRAG, Dense R-GCN, Dense-FT R-GCN |
| Execution provenance | ISETrace (canonical/graph/motif library only) | experiment integration pending |

The evidence workflow remains runnable. ISETrace has revision-pinned download provisioning plus canonical trajectory adaptation, query-independent provenance graph construction, and diverse template-query motif synthesis. It remains intentionally absent from the experiment dataset/config registry until split, retrieval, model, evaluation, and natural-query contracts are defined.

| Graph | Owner | Consumers |
|---|---|---|
| `EvidenceGraph` | dataset stage | evidence R-GCN only |
| GraphRAG entity graph | GraphRAG method (private) | GraphRAG only |
| `ProvenanceGraph` | canonical trajectory projector | motif synthesis only (currently) |

## Research boundary

- Flat methods are lexical/semantic baselines.
- GraphRAG tests method-owned entity propagation without a dataset graph artifact.
- Evidence R-GCN learns independent node scores on `EvidenceGraph`.
- The legacy synthetic-provenance EPGM and Provenance R-GCN implementations were removed rather than carried into ISETrace.
- The new provenance graph is derived only from canonical execution trajectories and does not contain query nodes, answer nodes, labels, or semantic edge weights.
- Namespaced graph kinds plus canonical source spans reserve a clean extension point for later NLP-derived claims, decisions, and semantic relations.
- Motif shape and query intent are separate; each supported pair has multiple style-tagged templates, but natural LLM/human test queries remain future work.

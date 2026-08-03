# Project overview

Graph structure should help recover complete evidence sets or execution paths beyond flat semantic retrieval.

## Domains

| Domain | Datasets | Methods |
|---|---|---|
| Evidence retrieval | HotpotQA, 2Wiki, MuSiQue | BM25, Dense, Dense-FT, GraphRAG, Dense R-GCN, Dense-FT R-GCN |
| Execution provenance | ISETrace natural-query pilot | BM25, Dense, GraphRAG, provenance path, Provenance R-GCN |

The evidence workflow remains runnable. The revision-pinned ISETrace corpus has canonical trajectory adaptation, query-independent provenance graphs, deterministic motif templates, and frozen trajectory-grouped train/dev/test ownership. Train/dev request exact natural/template task counts and may be template-only; test is always natural-only. The current generated natural-query corpus is explicitly unreviewed and is suitable for engineering validation, not final paper claims.

| Graph | Owner | Consumers |
|---|---|---|
| `EvidenceGraph` | dataset stage | evidence R-GCN only |
| GraphRAG entity graph | GraphRAG method (private) | GraphRAG only |
| `ProvenanceGraph` | canonical trajectory projector | motif synthesis + `provenance_path` + `provenance_rgcn` |

## Research boundary

- Flat methods are lexical/semantic baselines.
- GraphRAG is a deterministic retrieval-only FastGraphRAG adaptation: it builds a private noun-phrase co-occurrence graph over small text units, runs query-personalized PageRank, and projects graph scores back to the shared retrieval candidates. It never receives native provenance edges.
- Evidence R-GCN learns independent node scores on `EvidenceGraph`.
- `provenance_rgcn` is a new thin adapter over the maintained evidence R-GCN tensor/batching/optimizer/checkpoint stack; it does not restore the deleted label-conditioned provenance implementation or IDs.
- The new provenance graph is derived only from canonical execution trajectories and does not contain query nodes, answer nodes, labels, or semantic edge weights.
- Namespaced graph kinds plus canonical source spans reserve a clean extension point for later NLP-derived claims, decisions, and semantic relations.
- Motif shape and authoring intent remain internal task-planning signals only. Each task receives a deterministic style instruction, while the durable output is always the four-field v7 record; generated candidates remain provisional until manual acceptance.
- ISETrace split assignment groups every trajectory connected through a shared source intent or exact normalized intent text; no such task identity crosses train, dev, and test.

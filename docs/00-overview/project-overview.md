# Project overview

Graph structure should help recover complete evidence sets or execution paths beyond flat semantic retrieval.

## Domains

| Domain | Datasets | Methods |
|---|---|---|
| Evidence retrieval | HotpotQA, 2Wiki, MuSiQue | BM25, Dense, Dense-FT, GraphRAG, Dense R-GCN, Dense-FT R-GCN |
| Execution provenance | ISETrace natural-query pilot | BM25, Dense, GraphRAG, provenance path (non-training) |

The evidence workflow remains runnable. The complete revision-pinned ISETrace corpus has canonical trajectory adaptation, query-independent provenance graph construction, diverse motif/query synthesis, a fixed leakage-safe 80/10/10 intent-component split, and a test-only non-training retrieval workflow. The current 100-query natural pilot is explicitly unreviewed and is suitable for engineering validation, not final paper claims.

| Graph | Owner | Consumers |
|---|---|---|
| `EvidenceGraph` | dataset stage | evidence R-GCN only |
| GraphRAG entity graph | GraphRAG method (private) | GraphRAG only |
| `ProvenanceGraph` | canonical trajectory projector | motif synthesis + `provenance_path` only |

## Research boundary

- Flat methods are lexical/semantic baselines.
- GraphRAG tests method-owned text-entity propagation without access to native provenance edges; title-free ToolOutputs use bounded shared text entities.
- Evidence R-GCN learns independent node scores on `EvidenceGraph`.
- The legacy synthetic-provenance EPGM and Provenance R-GCN implementations were removed rather than carried into ISETrace.
- The new provenance graph is derived only from canonical execution trajectories and does not contain query nodes, answer nodes, labels, or semantic edge weights.
- Namespaced graph kinds plus canonical source spans reserve a clean extension point for later NLP-derived claims, decisions, and semantic relations.
- Motif shape and query intent are separate; each supported pair has multiple style-tagged templates. Natural LLM queries remain provisional until manual acceptance, and review/label policies are persisted in pilot artifacts.
- ISETrace split assignment groups every trajectory connected through a shared source intent or exact normalized intent text; no such task identity crosses train, dev, and test.

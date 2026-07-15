# Domain abstractions

The canonical behavior is fixed by [`execution-provenance-retrieval-domain-plan.md`](../10-plans/execution-provenance-retrieval-domain-plan.md).

`MethodInputSpec` states the exact request type, supported task families, and required artifact. `RetrievalCapabilities` states whether a method returns ranked nodes, a native edge trace, and whether it is trainable. Workflow code must consult these meanings rather than infer them from a generic “graph method” flag.

The three graph meanings remain separate:

1. `EvidenceGraph`: dataset-derived evidence relations used by R-GCN.
2. GraphRAG entity graph: method-private and rebuilt from candidates.
3. `ExecutionProvenanceGraph`: source-native execution/dataflow history carried by its request.

BM25 and Dense share one flat implementation across evidence and provenance families. GraphRAG shares one entity-search implementation but receives `GraphRAGRequest`. R-GCN and the provenance retriever have no cross-domain projection.

R-GCN performs independent node scoring with BCE training and full-ranking inference. The provenance retriever is intentionally nontrainable in this change; it selects semantic seeds and scores actual typed paths.

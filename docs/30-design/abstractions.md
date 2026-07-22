# Domain abstractions

The canonical behavior is fixed by [`execution-provenance-retrieval-domain-plan.md`](../10-plans/execution-provenance-retrieval-domain-plan.md).

The method Registry is intentionally small. Each method definition records only its public identifier, concrete request type, and supported retrieval families. Retrieval settings and builders assemble concrete methods. Prefect workflow scheduling remains explicit in `graph_memory/experiment/workflow.py`; lifecycle, training dependency, artifact, and ablation scheduling metadata are not duplicated in the Registry.

The three graph meanings remain separate:

1. `EvidenceGraph`: dataset-derived evidence relations used by R-GCN.
2. GraphRAG entity graph: method-private and rebuilt from candidates.
3. `ExecutionProvenanceGraph`: source-native execution/dataflow history carried by its request.

BM25 and Dense share one flat implementation across evidence and provenance families. GraphRAG shares one entity-search implementation but receives `GraphRAGRequest`. R-GCN and the provenance retriever have no cross-domain projection.

R-GCN performs independent node scoring with BCE training and full-ranking inference. The stateless provenance retriever selects semantic seeds and scores actual typed paths.

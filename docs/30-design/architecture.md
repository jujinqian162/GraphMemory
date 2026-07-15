# Architecture

The locked domain design is [`execution-provenance-retrieval-domain-plan.md`](../10-plans/execution-provenance-retrieval-domain-plan.md).

```text
dataset adapter
  -> concrete retrieval request
  -> Registry semantic validation
  -> concrete method builder
  -> retrieval method
  -> complete ranking plus optional native trace
```

## Ownership

- `graph_memory/datasets/` projects source records into consumer-specific requests.
- `graph_memory/contracts/graphs.py` owns the traditional `EvidenceGraph` artifact.
- `graph_memory/graphs/provenance/` owns native execution-provenance values and validation.
- `graph_memory/retrieval/requests/` owns the closed request union.
- `graph_memory/retrieval/methods/graphrag/` owns deterministic entity-graph assembly, linking, PPR, projection, and `GraphRAGTrace`; the Registry builder assembles the graph before method execution.
- `graph_memory/retrieval/methods/execution_provenance/` owns bounded alternative-path search, single-pass path scoring, invalidation, and `ExecutionProvenanceTrace`.
- `graph_memory/models/graph_retriever/` owns node-wise R-GCN training and inference.
- `graph_memory/registry/` owns public IDs, request/family/artifact compatibility, capabilities, settings, and builders.
- `graph_memory/experiment/` schedules stages from actual artifact dependencies.

Only the R-GCN paths require prebuilt `EvidenceGraph` artifacts. GraphRAG and the provenance retriever cannot cause that stage to be scheduled. No compatibility aliases translate one graph domain into another.

# Logging and run records

Each stage writes a typed YAML run summary adjacent to its primary artifact. Summaries record invocation identity, inputs, outputs, counts, timings, status, and retrieval provenance. Local files and `run_state.yaml` remain authoritative; MLflow mirrors the job for comparison.

Retrieval summaries record the public method ID, encoder/model source when applicable, device, task count, and latency. GraphRAG and execution-provenance results may additionally contain the closed `metadata.native_trace` union: `entity_search` or `execution_provenance`. Entity relations, evidence edges, and execution-provenance edges remain distinct trace types.

The EvidenceGraph construction stage is `evidence_graphs` and its script is `scripts/build_evidence_graphs.py`. It is absent from plans that select only flat retrieval or GraphRAG and do not otherwise require evidence-graph evaluation/training.

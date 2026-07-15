# Cross-dataset request design

Dataset adapters own source parsing and projection into consumer-specific requests. HotpotQA, 2Wiki, and MuSiQue project into `TextRankingRequest`, `GraphRAGRequest` through the shared text view, or evidence-specific build/ranking requests. They do not synthesize execution provenance.

Native execution-provenance adapters project source-visible histories into `ExecutionProvenanceRankingRequest` and may also expose flat text candidates for BM25/Dense or GraphRAG requests. The separately named `twowiki_provenance` adapter is an explicit synthetic-benchmark exception: a one-time converter constructs and audits the graph from labels before normal workflow preparation; standard 2Wiki never performs that projection at runtime.

The Registry validates the exact request type and task family before method execution. This preserves dataset/retriever decoupling without hiding incompatible graph meanings inside a universal request.

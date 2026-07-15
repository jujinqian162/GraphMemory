# Cross-dataset request design

Dataset adapters own source parsing and projection into consumer-specific requests. HotpotQA, 2Wiki, and MuSiQue project into `TextRankingRequest`, `GraphRAGRequest` through the shared text view, or evidence-specific build/ranking requests. They do not synthesize execution provenance.

TRAJECT-Bench projects its public catalog tools into `TextRankingRequest` and an optional `EvidenceGraphBuildRequest` view. This view represents catalog candidates and catalog-declared connections, not the gold execution trace. The ordered gold calls remain in the label artifact and are used only for evaluation. BM25 and Dense are the maintained fast baselines for this adapter.

Native execution-provenance adapters project source-visible histories into `ExecutionProvenanceRankingRequest` and may also expose flat text candidates for BM25/Dense or GraphRAG requests. They do not project gold-only histories into R-GCN.

The Registry validates the exact request type and task family before method execution. This preserves dataset/retriever decoupling without hiding incompatible graph meanings inside a universal request.

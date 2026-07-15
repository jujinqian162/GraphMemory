# Execution-Provenance Graph Memory

This repository implements two explicit retrieval domains. Traditional evidence datasets (HotpotQA, 2WikiMultiHopQA, and MuSiQue-Ans) support exactly six methods: BM25, Dense, Dense-FT, GraphRAG, Dense R-GCN, and Dense-FT R-GCN. Execution-provenance datasets support BM25, Dense, GraphRAG, the typed Execution-Provenance Retriever, and a separate provenance-native R-GCN.

The domains do not project into each other. `EvidenceGraph` is a dataset-derived artifact used only by the two R-GCN methods. GraphRAG builds a private entity graph from its request. `ExecutionProvenanceGraph` is native dataset input for the provenance retriever. The locked design is [`docs/10-plans/execution-provenance-retrieval-domain-plan.md`](docs/10-plans/execution-provenance-retrieval-domain-plan.md).

The separately named `twowiki_provenance` benchmark deterministically converts recoverable 2Wiki support chains into audited synthetic ToolCall/ToolOutput graphs for stateless beam and provenance R-GCN comparisons. See the [2Wiki provenance runbook](docs/40-operations/twowiki-provenance.md).

## Quick start

```powershell
uv run pytest -q
uv run python experiment/plan.py name=quick_valid_100 profile=quick
uv run python experiment/run.py name=quick_valid_100 profile=quick
uv run python experiment/status.py name=quick_valid_100
uv run python experiment/inspect.py kind=methods
```

Hydra YAML plus closed Pydantic models form the experiment configuration contract. Run artifacts live under `runs/<name>/`. See [`docs/40-operations/commands.md`](docs/40-operations/commands.md), [`docs/20-contracts/retrieval-contracts.md`](docs/20-contracts/retrieval-contracts.md), and [`docs/30-design/architecture.md`](docs/30-design/architecture.md).

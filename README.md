# Execution-Provenance Graph Memory

Two retrieval domains, no cross-projection:

| Domain | Datasets | Methods |
|---|---|---|
| Evidence | HotpotQA, 2Wiki, MuSiQue | BM25, Dense, Dense-FT, GraphRAG, Dense R-GCN, Dense-FT R-GCN |
| Provenance | ISETrace (raw download only) | redesign pending |

`EvidenceGraph` feeds only the evidence R-GCN methods. GraphRAG builds a private entity graph. The legacy execution-provenance contracts and methods have been removed; a new ISETrace adapter will define its own trajectory-native contracts. Design: [`docs/30-design/architecture.md`](docs/30-design/architecture.md).

## Quick start

```powershell
uv run pytest -q
uv run python experiment/run.py name=quick_bm25 profile=quick method=bm25
uv run python experiment/inspect.py kind=methods
```

One Hydra job = one method (optional one R-GCN variant) = one Prefect Flow = one MLflow run. Scientific assets reuse under `data/processed/`; `runs/<name>/` is output-only.

- Commands: [`docs/40-operations/commands.md`](docs/40-operations/commands.md)
- Retrieval matrix: [`docs/20-contracts/retrieval-contracts.md`](docs/20-contracts/retrieval-contracts.md)
- Doc index: [`docs/README.md`](docs/README.md)

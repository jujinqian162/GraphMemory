# Execution-Provenance Graph Memory

Two retrieval domains, no cross-projection:

| Domain | Datasets | Methods |
|---|---|---|
| Evidence | HotpotQA, 2Wiki, MuSiQue | BM25, Dense, Dense-FT, GraphRAG, Dense R-GCN, Dense-FT R-GCN |
| Provenance | ISETrace natural-query pilot | BM25, Dense, GraphRAG, provenance path (all non-training) |

`EvidenceGraph` feeds only the evidence R-GCN methods. GraphRAG builds a private text-derived entity graph. ISETrace provides canonical ordered trajectories, query-independent provenance graphs, schema-derived queries, and a test-only non-training workflow. BM25, Dense, and GraphRAG receive identical ToolOutput text candidates; `provenance_path` alone consumes the native graph and completes bounded logical dependencies without training. The committed 100-query config explicitly permits unreviewed records for engineering pilot runs and is not formal paper gold. See [`docs/40-operations/isetrace-nontrain-retrieval.md`](docs/40-operations/isetrace-nontrain-retrieval.md), [`docs/40-operations/isetrace-split.md`](docs/40-operations/isetrace-split.md), and [`docs/40-operations/isetrace-query-authoring.md`](docs/40-operations/isetrace-query-authoring.md).

## Quick start

```powershell
uv run pytest -q
uv run python experiment/run.py name=quick_bm25 profile=quick method=bm25
uv run python experiment/run.py name=isetrace_bm25 dataset=isetrace profile=smoke method=bm25 device=cpu
uv run python experiment/inspect.py kind=methods
```

One Hydra job = one method (optional one R-GCN variant) = one Prefect Flow = one MLflow run. Scientific assets reuse under `data/processed/`; `runs/<name>/` is output-only.

- Commands: [`docs/40-operations/commands.md`](docs/40-operations/commands.md)
- Retrieval matrix: [`docs/20-contracts/retrieval-contracts.md`](docs/20-contracts/retrieval-contracts.md)
- Doc index: [`docs/README.md`](docs/README.md)

# Execution-Provenance Graph Memory

Two retrieval domains, no cross-projection:

| Domain | Datasets | Methods |
|---|---|---|
| Evidence | HotpotQA, 2Wiki, MuSiQue | BM25, Dense, Dense-FT, GraphRAG, Dense R-GCN, Dense-FT R-GCN |
| Provenance | ISETrace natural-query pilot | BM25, Dense, Dense-FT, GraphRAG, provenance path, Provenance R-GCN |

`EvidenceGraph` feeds only the evidence R-GCN methods. GraphRAG builds a private text-derived entity graph. ISETrace provides canonical ordered trajectories, query-independent `ProvenanceGraph` records, trajectory-grouped natural-query splits, and deterministic train/dev template supervision. BM25, Dense, Dense-FT, and GraphRAG share the flat trajectory-chunk boundary; ISETrace Dense-FT maps exact spans to flat positives, uses text-only negatives, and receives no graph. `provenance_path` and `provenance_rgcn` consume native provenance content candidates. `provenance_rgcn` reuses the maintained R-GCN encoder/training/checkpoint stack and evaluates only the natural test split. Generated natural queries remain unreviewed engineering data until separately reviewed and frozen, so no formal paper result is claimed. See [`docs/40-operations/isetrace-dense-ft.md`](docs/40-operations/isetrace-dense-ft.md), [`docs/40-operations/isetrace-provenance-rgcn.md`](docs/40-operations/isetrace-provenance-rgcn.md), [`docs/40-operations/isetrace-nontrain-retrieval.md`](docs/40-operations/isetrace-nontrain-retrieval.md), and [`docs/40-operations/isetrace-query-authoring.md`](docs/40-operations/isetrace-query-authoring.md).

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

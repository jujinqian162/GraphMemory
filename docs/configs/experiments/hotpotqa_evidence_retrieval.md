# Default evidence experiment

Each Hydra job composes one of the six evidence methods: BM25, Dense, Dense-FT, GraphRAG, Dense R-GCN, or Dense-FT R-GCN. The governing scientific design is [`execution-provenance-retrieval-domain-plan.md`](../../10-plans/execution-provenance-retrieval-domain-plan.md).

```powershell
uv run python experiment/run.py name=hotpotqa_bm25 dataset=hotpotqa profile=quick method=bm25
uv run python experiment/run.py -m name=hotpotqa_baselines dataset=hotpotqa profile=quick method=bm25,dense,graphrag
```

Hydra groups and overrides control dataset, profile, one method, seed, device, top-k, cache refresh, benchmark, and method-specific settings. Resolved values are projected under the named output-only run; reusable scientific assets live under `data/processed/`.

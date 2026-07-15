# Default evidence experiment

`configs/config.yaml` composes the six-method evidence matrix: BM25, Dense, Dense-FT, GraphRAG, Dense R-GCN, and Dense-FT R-GCN. The governing design is [`execution-provenance-retrieval-domain-plan.md`](../../10-plans/execution-provenance-retrieval-domain-plan.md).

```powershell
uv run python experiment/plan.py name=hotpotqa_quick dataset=hotpotqa profile=quick
uv run python experiment/run.py name=hotpotqa_quick dataset=hotpotqa profile=quick
```

Hydra groups and overrides control dataset, profile, selected methods, seed, device, top-k, stage bounds, cache, ablations, and method-specific settings. Resolved values are persisted under the named run.

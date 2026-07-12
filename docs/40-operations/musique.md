# MuSiQue operations

The repository supports the answerable MuSiQue split with paragraph-level candidates and decomposition-derived dependency edges. Use the `musique` dataset group; split sources and capacities are declared in `configs/dataset/musique.yaml`.

```powershell
uv run python experiment/plan.py `
  name=musique_smoke dataset=musique profile=smoke device=cpu `
  'methods=[bm25,bm25_graph_rerank]'

uv run python experiment/run.py `
  name=musique_smoke dataset=musique profile=smoke device=cpu `
  'methods=[bm25,bm25_graph_rerank]'

uv run python experiment/status.py name=musique_smoke
```

MuSiQue raw files are JSONL. Preparation rejects unanswerable or malformed records, keeps gold answers/support/decomposition out of ranking inputs, and writes labels separately. Graph construction and retrieval consume only projected ranking requests. Evaluation joins predictions with labels after retrieval and reports paragraph evidence metrics plus dependency path metrics for graph-aware methods.

All other supported methods can be selected with the same `methods=[...]` override. Dense and trainable methods require their configured model assets and device.

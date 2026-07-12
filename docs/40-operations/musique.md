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

## R-GCN beam comparison

Both existing R-GCN method ids now use beam decoding in place; no beam-specific method is registered. Compare the bounded beam matrix with separate fresh run names so training and inference use the same width:

```powershell
uv run python experiment/run.py name=musique_rgcn_beam1 profile=full device=cuda `
  'methods=[dense_ft_rgcn_graph_retriever]' `
  method_configs.dense_ft_rgcn_graph_retriever.train.beam.training_beam_size=1 `
  method_configs.dense_ft_rgcn_graph_retriever.train.beam.inference_beam_size=1

# Repeat with fresh names and both values set to 2, then 4.
```

Keep each run's training and inference widths equal; crossed pairs are rejected by typed validation. Select primarily on Full Support@5 and retain Full Support@10, path metrics, latency, and seed stability as guardrails. R-GCN checkpoints written before this beam-decoder change are not loadable and must be retrained.

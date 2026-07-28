# Experiment commands

One command runs one final method and, for R-GCN, one variant. Hydra composes config; Prefect runs one Flow; MLflow records one run.

## Single jobs

```powershell
uv run python experiment/run.py name=hotpot_bm25 dataset=hotpotqa profile=quick method=bm25

uv run python experiment/run.py `
  name=hotpot_rgcn dataset=hotpotqa profile=smoke device=cpu `
  method=dense_rgcn_graph_retriever method.variant=full_rgcn

uv run python experiment/run.py `
  name=provenance_rgcn dataset=twowiki_provenance profile=smoke device=cpu `
  method=execution_provenance_rgcn_retriever method.variant=wo_graph
```

`cache.refresh=true` forces every reusable Task to re-execute; it is not part of scientific cache identity.

## Multirun and multi-GPU

```powershell
uv run python experiment/run.py -m `
  name=hotpot_baselines dataset=hotpotqa profile=quick `
  method=bm25,dense,graphrag

# One variant per GPU; no in-Flow training fan-out
uv run python experiment/run.py name=rgcn_full dataset=hotpotqa profile=quick device=cuda:0 method=dense_rgcn_graph_retriever method.variant=full_rgcn
uv run python experiment/run.py name=rgcn_wo_graph dataset=hotpotqa profile=quick device=cuda:1 method=dense_rgcn_graph_retriever method.variant=wo_graph

# One R-GCN job; frozen E5 encoding uses all CUDA devices visible to PyTorch
uv run python experiment/run.py name=rgcn_multi_encode dataset=hotpotqa profile=quick device=cuda:0 encoding.enable_gpupool=true method=dense_rgcn_graph_retriever
```

Evidence and provenance R-GCN jobs persist train/dev float32 embeddings as a
separate Prefect Task under `data/processed/frozen_embeddings/`. The
`encoding.enable_gpupool` flag and `encoding.chunk_size` control only runtime
placement and streaming. With the flag enabled, all CUDA devices reported by
`torch.cuda.device_count()` are used; otherwise the root `device` is used.
`CUDA_VISIBLE_DEVICES` may still be set externally to limit which cards PyTorch
can see. Runtime settings do not invalidate scientific cache.

## Inspect and deliver

```powershell
uv run python experiment/inspect.py kind=methods
uv run python experiment/inspect.py kind=profiles
uv run python experiment/inspect.py kind=variants
uv run python scripts/deliver/collect_run_artifacts.py --name hotpot_baselines
```

Collector mirrors small output-only files to `results/<name>/` and records asset digests without copying `data/processed/`.

## Verification

```powershell
uv run pytest -q
uv run ruff check .
uv run basedpyright --level error
uv run python -m compileall -q graph_memory experiment scripts tests
git diff --check
```

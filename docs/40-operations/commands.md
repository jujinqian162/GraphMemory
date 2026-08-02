# Experiment commands

One command runs one final method and, for R-GCN, one variant. Hydra composes config; Prefect runs one Flow; MLflow records one run.

## Single jobs

```powershell
uv run python experiment/run.py name=hotpot_bm25 dataset=hotpotqa profile=quick method=bm25

uv run python experiment/run.py `
  name=hotpot_rgcn dataset=hotpotqa profile=smoke device=cpu `
  method=dense_rgcn_graph_retriever method.variant=full_rgcn
```

`cache.refresh=true` forces every reusable Task to re-execute; it is not part of scientific cache identity.

## ISETrace non-training pilot

```powershell
uv run python experiment/run.py name=isetrace_bm25_smoke dataset=isetrace profile=smoke method=bm25 device=cpu
uv run python experiment/run.py name=isetrace_path_smoke dataset=isetrace profile=smoke method=provenance_path device=cpu

uv run python experiment/run.py -m `
  name=isetrace_nontrain_pilot dataset=isetrace profile=full device=cuda:0 `
  method=bm25,dense,graphrag,provenance_path
```

## ISETrace Provenance R-GCN engineering run

```powershell
uv run python experiment/run.py name=isetrace_rgcn_smoke dataset=isetrace profile=smoke method=provenance_rgcn device=cpu
uv run python experiment/run.py name=isetrace_rgcn_no_graph dataset=isetrace profile=quick method=provenance_rgcn method.variant=wo_graph device=cuda:0
```

The trainable lifecycle prepares mixed train/dev pools, builds candidate pairs, freezes graph/query embeddings, selects on natural dev Recall@5, reloads the strict checkpoint, and evaluates the natural-only test split. `profile=full` does not apply the evidence benchmark's fixed dev cap to ISETrace.

The committed generated queries are unreviewed engineering inputs only. See [`isetrace-provenance-rgcn.md`](isetrace-provenance-rgcn.md) and [`isetrace-nontrain-retrieval.md`](isetrace-nontrain-retrieval.md) before interpreting metrics.

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

R-GCN jobs persist train/dev float32 embeddings as a
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

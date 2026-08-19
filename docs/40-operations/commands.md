# Experiment commands

One command runs one final method and one resolved variant where supported. Dense/Dense-FT/Cross-Encoder use `flat|provenance_unit`; R-GCN uses its ablation variants. Hydra composes config; Prefect runs one Flow; MLflow records one run. Every new command must follow the canonical [`name=` convention](run-naming.md); older examples below may retain historical names only when reproducing an existing run.

## Single jobs

```powershell
uv run python experiment/run.py name=hotpot_bm25 dataset=hotpotqa profile=quick method=bm25

uv run python experiment/run.py `
  name=hotpot_rgcn dataset=hotpotqa profile=smoke device=cuda:0 `
  method=dense_rgcn_graph_retriever method.variant=full_rgcn
```

`cache.refresh=true` forces every reusable Task to re-execute; it is not part of scientific cache identity.

## ISETrace non-training pilot

```powershell
uv run python experiment/run.py name=isetrace_bm25_smoke dataset=isetrace profile=smoke method=bm25 device=cuda:0
uv run python experiment/run.py name=isetrace_provenance_unit_dense_smoke dataset=isetrace profile=smoke method=dense method.variant=provenance_unit device=cuda:0
uv run python experiment/run.py name=isetrace_provenance_unit_dense_ft_smoke dataset=isetrace profile=smoke method=dense_ft method.variant=provenance_unit device=cuda:0
uv run python experiment/run.py name=isetrace_cross_encoder_flat_smoke dataset=isetrace profile=smoke method=cross_encoder method.variant=flat device=cuda:0
uv run python experiment/run.py name=isetrace_cross_encoder_pu_smoke dataset=isetrace profile=smoke method=cross_encoder method.variant=provenance_unit device=cuda:0
uv run python experiment/run.py name=isetrace_path_smoke dataset=isetrace profile=smoke method=provenance_path device=cuda:0

uv run python experiment/run.py -m `
  name=isetrace_nontrain_pilot dataset=isetrace profile=full device=cuda:0 `
  method=bm25,dense,graphrag,provenance_path
```

## ISETrace Provenance R-GCN engineering run

```powershell
uv run python experiment/run.py name=isetrace_rgcn_smoke dataset=isetrace profile=smoke method=provenance_rgcn device=cuda:0
uv run python experiment/run.py name=isetrace_rgcn_no_graph dataset=isetrace profile=quick method=provenance_rgcn method.variant=wo_graph device=cuda:0
uv run python experiment/run.py name=isetrace_pu_dense_ft_rgcn_full dataset=isetrace profile=full method=provenance_unit_dense_ft_rgcn method.variant=full_rgcn device=cuda:0
uv run python experiment/run.py name=isetrace_pu_dense_ft_rgcn_wo_graph dataset=isetrace profile=full method=provenance_unit_dense_ft_rgcn method.variant=wo_graph device=cuda:1
```

The seeded config composes the canonical `dense_ft variant=provenance_unit` stage. Its pair and checkpoint Task inputs are identical to the standalone baseline, so Prefect reuses that checkpoint across `full_rgcn`, `wo_graph`, and the E2 controls. `wo_graph` returns the cached seed's cosine scores exactly; `full_rgcn` adds a zero-initialized learned graph residual and dev selection may retain epoch 0 when training does not improve the seed. E2 variants are `homogeneous_gcn`, `wo_feeds`, `wo_execution_ownership`, `wo_artifact_io`, `wo_chunk_adjacency`, and `random_edges`; each changes only its checkpointed message relation/topology policy. The remaining trainable lifecycle builds provenance candidate pairs, reloads the strict checkpoint, and evaluates the natural-only test split. `profile=full` does not apply the evidence benchmark's fixed dev cap to ISETrace.

## ISETrace Training-Size Robustness

Training size is controlled by the existing trajectory count, not by a task-row cap. With fixed `split_seed=13`, smaller train counts select nested prefixes while the 352-trajectory dev and 1,207-trajectory test partitions remain unchanged. Keep `profile=full` so every condition evaluates the same 580 dev tasks and 2,000 test tasks.

```bash
PERCENTAGES=(10 25 50 100)
TRAIN_TRAJECTORIES=(241 603 1205 2410)
SEEDS=(13 17 29 37 41)
DEVICE=cuda:0

for index in "${!PERCENTAGES[@]}"; do
  percentage="${PERCENTAGES[$index]}"
  train_trajectories="${TRAIN_TRAJECTORIES[$index]}"
  for seed in "${SEEDS[@]}"; do
    common=(
      dataset=isetrace
      profile=full
      "device=${DEVICE}"
      split_seed=13
      "seed=${seed}"
      "dataset.trajectories.splits.train=${train_trajectories}"
    )

    uv run python experiment/run.py \
      "name=isetrace_v7_trainsize${percentage}_dft_pu_s${seed}" \
      "${common[@]}" method=dense_ft method.variant=provenance_unit

    uv run python experiment/run.py \
      "name=isetrace_v7_trainsize${percentage}_rgcn_nograph_s${seed}" \
      "${common[@]}" method=provenance_unit_dense_ft_rgcn \
      method.variant=wo_graph

    uv run python experiment/run.py \
      "name=isetrace_v7_trainsize${percentage}_rgcn_full_s${seed}" \
      "${common[@]}" method=provenance_unit_dense_ft_rgcn \
      method.variant=full_rgcn
  done
done
```

The four conditions correspond to 10%, 25%, 50%, and 100% of the 2,410 training trajectories, rounded to whole trajectories. They intentionally keep the one-epoch Dense-FT and 15-epoch residual recipes unchanged; this is fixed-recipe robustness, not per-size hyperparameter tuning. The standalone PU Dense-FT, exact `wo_graph` passthrough, and residual runs share content-addressed preparation and Dense checkpoint artifacts whenever their scientific inputs match.

After all 60 runs complete, validate the frozen cohort and aggregate five-seed means, sample standard deviations, and residual-versus-exact-seed trajectory-cluster bootstrap intervals:

```bash
uv run python scripts/analyze_isetrace_training_size.py \
  --run-root runs \
  --output results/isetrace/training-size/analysis.json \
  --output-csv results/isetrace/training-size/analysis.csv \
  --report results/isetrace/training-size/report.md \
  --figure results/isetrace/training-size/training-size.pdf
```

The analysis rejects missing seeds, incorrect method variants, changed split counts or split seed, non-2,000-query test outputs, and inconsistent test artifacts or trajectory clusters before reporting results.

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

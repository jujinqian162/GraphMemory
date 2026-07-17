# Experiment commands

One command runs one final method and, where supported, one singular R-GCN variant. The command composes Hydra configuration, opens one MLflow run, and executes one synchronous Prefect Flow. There is no plan/status/reset command, stage range, or generated stage CLI.

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

Use `cache.refresh=true` to force every reusable Task in the selected Flow to execute again. The option is operational and does not become part of scientific cache identity.

## Baseline sweep

Hydra multirun creates one process, Prefect Flow run, MLflow run, and deterministic child output directory per method:

```powershell
uv run python experiment/run.py -m `
  name=hotpot_baselines dataset=hotpotqa profile=quick `
  method=bm25,dense,graphrag
```

## Ablations and multiple GPUs

Variants are independent jobs; the Flow does not accept a variant list and does not call `task.submit()`. Launch one command per GPU so CUDA ownership and failures remain isolated while compatible prepared/graph/pair assets are reused through the shared Prefect cache:

```powershell
uv run python experiment/run.py name=rgcn_ablation_full dataset=hotpotqa profile=quick device=cuda:0 method=dense_rgcn_graph_retriever method.variant=full_rgcn
uv run python experiment/run.py name=rgcn_ablation_wo_graph dataset=hotpotqa profile=quick device=cuda:1 method=dense_rgcn_graph_retriever method.variant=wo_graph
```

For Hydra-managed sequential or externally parallel jobs, keep a shared user-visible study name and vary the deterministic job/output selector.

## Inspect and deliver

```powershell
uv run python experiment/inspect.py kind=methods
uv run python experiment/inspect.py kind=profiles
uv run python experiment/inspect.py kind=variants
uv run python scripts/deliver/collect_run_artifacts.py --name hotpot_baselines
```

The collector mirrors all small output-only files to `results/<name>/`, including every multirun child. It records processed asset URIs and digests from each `assets/manifest.yaml` without copying `data/processed/`.

## Verification

```powershell
uv run pytest -q
uv run ruff check .
uv run basedpyright --level error
uv run python -m compileall -q graph_memory experiment scripts tests
openspec validate replace-experiment-runner-with-prefect-workflow --strict
git diff --check
```

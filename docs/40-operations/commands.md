# Experiment commands

Date: 2026-07-11

`plan` and `run` compose `configs/config.yaml` with Hydra. `status`, `inspect`, and `reset` parse only their closed `key=value` command models and create no Hydra job.

## Plan and run

```powershell
uv run python experiment/plan.py name=quick_valid_100 profile=quick
uv run python experiment/run.py name=quick_valid_100 profile=quick
```

The default config selects the approved seven-method HotpotQA workflow. Select methods, dataset, device, stage bounds, or cache behavior with Hydra overrides:

```powershell
uv run python experiment/plan.py `
  name=twowiki_smoke dataset=2wiki profile=smoke device=cpu `
  'methods=[bm25,bm25_graph_rerank]' stages.from=graphs stages.to=evaluate

uv run python experiment/run.py `
  name=dense_ft_only profile=quick device=cuda `
  'methods=[dense_ft]' cache.enabled=false
```

An existing name may be reopened only with the identical resolved configuration and run mode. Use a new name or the reset command for a different configuration.

Former root presets are ordinary copyable overrides:

```powershell
# 2Wiki tiny
uv run python experiment/run.py name=twowiki_tiny dataset=2wiki profile=tiny device=cpu `
  'methods=[bm25,dense,dense_graph_rerank,dense_rgcn_graph_retriever,dense_ft,dense_ft_rgcn_graph_retriever]'

# Complete HotpotQA dev window as test
uv run python experiment/run.py name=hotpotqa_dev_full profile=cloud-full `
  dataset.splits.test.offset=0

# Memory Stream's importance-backed inputs; method/profile remain independent
uv run python experiment/run.py name=memory_stream dataset=hotpotqa-memory-stream `
  profile=memory-full 'methods=[bm25,dense,memory_stream]'
```

## Status, inspection, and reset

```powershell
uv run python experiment/status.py name=quick_valid_100
uv run python experiment/inspect.py kind=methods
uv run python experiment/inspect.py kind=profiles
uv run python experiment/inspect.py kind=ablations
uv run python experiment/reset.py name=quick_valid_100
```

`status` derives `missing`, `complete`, `stale`, and `alias` from local artifacts plus matching typed stage summaries. MLflow state is never used as cache truth. `reset` is the only public destructive named-run operation and refuses paths outside the repository-owned run root.

## Sequential multirun

Hydra's BasicLauncher runs jobs sequentially. Each job has an independent typed state and MLflow parent:

```powershell
uv run python experiment/run.py -m `
  name=seed_sweep profile=smoke device=cpu 'methods=[bm25]' seed=13,17
```

Jobs are stored as `runs/seed_sweep/<job-number>_<override-dirname>/`. Pass `job=<directory-name>` to `status` or `reset` when more than one job exists.

## R-GCN ablations

```powershell
uv run python experiment/plan.py `
  name=rgcn_ablation_cloud profile=cloud-full `
  'methods=[dense_rgcn_graph_retriever]' ablation.variants=all

uv run python experiment/run.py `
  name=rgcn_ablation_cloud profile=cloud-full `
  'methods=[dense_rgcn_graph_retriever]' ablation.variants=all
```

`full_rgcn` aliases the ordinary R-GCN result. Model variants reuse ordinary pair artifacts; `wo_hard_negatives` owns variant-specific pairs. With `ablation.only=true`, the ordinary baseline metric must already be complete under the same run identity.

## Direct stage debugging

Every low-level stage accepts exactly one resolved YAML contract:

```powershell
uv run python scripts/prepare_hotpotqa.py --config runs/quick_valid_100/config/stages/prepare/train.yaml
uv run python scripts/build_graphs.py --config runs/quick_valid_100/config/stages/graphs/train.yaml
uv run python scripts/build_train_pairs.py --config runs/quick_valid_100/config/stages/pairs/dense_rgcn_graph_retriever.yaml
uv run python scripts/train_method.py --config runs/quick_valid_100/config/stages/train/dense_ft.yaml
uv run python scripts/run_retrieval.py --config runs/quick_valid_100/config/stages/retrieve/dense_rgcn_graph_retriever.yaml
uv run python scripts/evaluate_retrieval.py --config runs/quick_valid_100/config/stages/evaluate/dense_rgcn_graph_retriever.yaml
uv run python scripts/aggregate_tables.py --config runs/quick_valid_100/config/stages/aggregate/aggregate.yaml
```

Do not add scientific flags to these commands. Change Hydra source groups or root overrides, plan with a new name, and use the generated YAML. A directly invoked stage writes local outputs and its YAML summary but creates no MLflow run.

## Old-to-new mapping

| Retired operation | Current operation |
| --- | --- |
| initialize then plan | `python experiment/plan.py name=<name> ...` |
| initialize then run | `python experiment/run.py name=<name> ...` |
| positional status | `python experiment/status.py name=<name>` |
| config/method/profile listing | `python experiment/inspect.py kind=<kind>` |
| forced reinitialization | `python experiment/reset.py name=<name>`, then plan or run |
| disable cache flag | `cache.enabled=false` |
| stage range flags | `stages.from=<stage> stages.to=<stage>` |
| method CSV flag | `'methods=[bm25,dense]'` |

The retired positional CLI, JSON configuration trees, custom run roots, force behavior, and parameter-style stage commands have no compatibility adapters.

## MLflow UI

Tracking defaults use one repository-local SQLite database and artifact root. Start the UI from the repository root:

```powershell
uv run mlflow ui `
  --backend-store-uri sqlite:///runs/.mlflow/tracking.db `
  --default-artifact-root ./runs/.mlflow/artifacts
```

Open `http://127.0.0.1:5000`. Each Hydra job owns one parent run; each executed stage attempt owns one child. Cache hits create no child. Local outputs, `run_state.yaml`, and adjacent `*.run_summary.yaml` files remain the scientific source of truth.

## Delivery and verification

```powershell
uv run python scripts/deliver/collect_run_artifacts.py --name quick_valid_100
uv run pytest -q
uv run ruff check .
uv run basedpyright --level error
uv run python -m compileall -q graph_memory scripts tests
openspec validate refactor-experiment-config-workflow --strict
git diff --check
```

Delivery reads typed run state and copies resolved YAML, stage summaries, aggregate tables, metrics, selected tuning files, and compact failure artifacts. It excludes datasets, graphs, predictions, train pairs, checkpoints, and model directories.

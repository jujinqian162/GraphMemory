# Experiment commands

Date: 2026-07-12

`plan` and `run` compose `configs/config.yaml` with Hydra. `status`, `inspect`, and `reset` parse only their closed `key=value` command models and create no Hydra job.

## Plan and run

```powershell
uv run python experiment/plan.py name=quick_valid_100 profile=quick
uv run python experiment/run.py name=quick_valid_100 profile=quick
```

The default config selects the approved seven-method MuSiQue full workflow. Select methods, dataset, profile, device, stage bounds, or cache behavior with Hydra overrides:

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
# 2Wiki smoke
uv run python experiment/run.py name=twowiki_smoke dataset=2wiki profile=smoke device=cpu `
  'methods=[bm25,dense,dense_graph_rerank,dense_rgcn_graph_retriever,dense_ft,dense_ft_rgcn_graph_retriever]'

# Full HotpotQA dataset
uv run python experiment/run.py name=hotpotqa_full dataset=hotpotqa profile=full `
  dataset.splits.test.offset=0

# Memory Stream's importance-backed inputs; method/profile remain independent
uv run python experiment/run.py name=memory_stream dataset=hotpotqa-memory-stream `
  profile=full 'methods=[bm25,dense,memory_stream]'
```

## Status, inspection, and reset

```powershell
uv run python experiment/status.py name=quick_valid_100
uv run python experiment/inspect.py kind=methods
uv run python experiment/inspect.py kind=profiles
uv run python experiment/inspect.py kind=ablations
uv run python experiment/inspect.py kind=jobs name=rgcn_layers
uv run python experiment/status.py name=rgcn_layers job=0_num_layers=2
uv run python experiment/reset.py name=rgcn_layers job=0_num_layers=2
uv run python experiment/reset.py name=quick_valid_100
```

`status` derives `missing`, `complete`, `stale`, and `alias` from local artifacts plus matching typed stage summaries. MLflow state is never used as cache truth. `inspect kind=jobs` lists concise multirun selectors. `reset` is the only public destructive run operation; it can remove a whole name or one exact `job=...` child and refuses paths outside the repository-owned run root.

## Sequential multirun

Hydra's BasicLauncher runs jobs sequentially. Each job has an independent typed state and MLflow parent:

```powershell
uv run python experiment/run.py -m `
  name=rgcn_layers profile=smoke device=cpu `
  'methods=[dense_rgcn_graph_retriever]' `
  method_configs.dense_rgcn_graph_retriever.train.model.num_layers=2,3,4
```

Jobs are stored as `runs/rgcn_layers/0_num_layers=2`, `1_num_layers=3`, and `2_num_layers=4`. Fixed identity overrides such as `name`, `dataset`, `profile`, and `methods` do not expand every leaf. `multirun.yaml` catalogs the sweep, while every job retains its complete `config/resolved.yaml` and `config/overrides.yaml`. Use the exact concise selector with `status` or `reset`, and list selectors with `inspect kind=jobs name=rgcn_layers`.

## R-GCN ablations

R-GCN training and checkpoint-backed retrieval use the same configured beam width. The default is beam `2` with at most five unique evidence selections and set-level deduplication. A learned `STOP` action ends the selected prefix; base R-GCN logits complete the ranking. Existing pre-beam R-GCN checkpoints are incompatible and require retraining.

```powershell
uv run python experiment/plan.py `
  name=rgcn_ablation_full profile=full `
  'methods=[dense_rgcn_graph_retriever]' ablation.enable=true

uv run python experiment/run.py `
  name=rgcn_ablation_full profile=full `
  'methods=[dense_rgcn_graph_retriever]' ablation.enable=true
```

`ablation.enable=true` runs every variant explicitly listed in the default
`ablation.variants` list. To run a subset, override that list, for example with
`ablation.variants='[wo_bridge,wo_graph]'`.

The valid variant values are `wo_bridge`, `wo_entity_overlap`, `wo_sequential`,
`wo_query_overlap`, `wo_graph`, `wo_edge_type`, `wo_edge_weight`, `wo_seed_score`,
and `wo_hard_negatives`. `full_rgcn` aliases the ordinary R-GCN result and is not
a selectable variant. Model variants reuse ordinary pair artifacts, while
`wo_hard_negatives` owns variant-specific pairs.

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

Open `http://127.0.0.1:5000`. Each Hydra job owns one parent summary run. Its Overview contains the readable all-baseline result table when aggregation is available; `config/`, `results/`, and `workflow/` contain job-level artifacts. MLflow 3.14 stores this table in `mlflow.note.content`; plain-text Markdown rendering is accepted, and the CSVs under `results/` remain the durable fallback.

Each user-selected baseline, including each executable ablation variant, owns one child. Search child parameters such as `train.trainer.epochs`, `encoder.model_name`, or `scoring.*` without a `method_configs.<method>` prefix. Common `final.*` metrics make children directly comparable, and trainable children additionally expose epoch-indexed `train.*` series. Shared prepare/graph/aggregate stages do not create children. Cached and resumed stages populate or reuse the same baseline child.

To compare methods or multirun jobs in MLflow:

1. Open the `graph-memory` experiment and filter `tags.graph_memory.run_kind = baseline`.
2. Select baseline children from the desired parents/jobs.
3. Choose Compare Runs and use the shared `final.*` columns; trainable curves use `train.*`.

Parent runs intentionally have no native result metrics, training series, saved chart configuration, or repository-generated comparison image. Counts, timings, statuses, errors, and artifact sizes remain in tags and YAML summaries. Local outputs, `run_state.yaml`, adjacent `*.run_summary.yaml`, and result CSVs remain the scientific source of truth.

The cutover is direct. Existing SQLite rows remain untouched historical records. Do not reuse a local name created by the former stage-child organization: choose a fresh name or run the normal explicit reset first. There is no compatibility reader, dual-write path, old-run migration, or database rewrite.

## Delivery and verification

```powershell
uv run python scripts/deliver/collect_run_artifacts.py --name quick_valid_100
uv run pytest -q
uv run ruff check .
uv run basedpyright --level error
uv run python -m compileall -q graph_memory scripts tests
openspec validate reorganize-mlflow-experiment-tracking --strict
git diff --check
```

Delivery reads typed run state and copies resolved YAML, stage summaries, aggregate tables, metrics, selected tuning files, and compact failure artifacts. It excludes datasets, graphs, predictions, train pairs, checkpoints, and model directories.

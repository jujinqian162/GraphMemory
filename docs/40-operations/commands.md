# Experiment commands

The maintained experiment matrix and workflow constraints come from [`execution-provenance-retrieval-domain-plan.md`](../10-plans/execution-provenance-retrieval-domain-plan.md).

## Plan and run

```powershell
uv run python experiment/plan.py name=quick_valid_100 profile=quick
uv run python experiment/run.py name=quick_valid_100 profile=quick

uv run python experiment/run.py `
  name=twowiki_smoke dataset=2wiki profile=smoke device=cpu `
  'methods=[bm25,dense,graphrag]'

uv run python experiment/run.py `
  name=rgcn_smoke profile=smoke device=cpu `
  'methods=[dense_rgcn_graph_retriever]'

uv run python experiment/run.py `
  name=traject_bench_bm25_smoke dataset=traject_bench profile=smoke device=cpu `
  'methods=[bm25]'
```

The default evidence workflow selects BM25, Dense, Dense-FT, GraphRAG, Dense R-GCN, and Dense-FT R-GCN. TRAJECT-Bench has an explicit dataset profile for retrieval-only BM25 and Dense baselines; see [`traject-bench.md`](traject-bench.md) for data download and server commands. There is no dataset profile that emits a native execution-provenance request yet.

## Inspect and control

```powershell
uv run python experiment/status.py name=quick_valid_100
uv run python experiment/inspect.py kind=methods
uv run python experiment/inspect.py kind=profiles
uv run python experiment/inspect.py kind=ablations
uv run python experiment/reset.py name=quick_valid_100
```

## Direct typed-stage debugging

```powershell
uv run python scripts/prepare_hotpotqa.py --config runs/<name>/config/stages/prepare/train.yaml
uv run python scripts/build_evidence_graphs.py --config runs/<name>/config/stages/evidence_graphs/train.yaml
uv run python scripts/build_train_pairs.py --config runs/<name>/config/stages/pairs/dense_rgcn_graph_retriever.yaml
uv run python scripts/train_method.py --config runs/<name>/config/stages/train/dense_rgcn_graph_retriever.yaml
uv run python scripts/run_retrieval.py --config runs/<name>/config/stages/retrieve/graphrag.yaml
uv run python scripts/evaluate_retrieval.py --config runs/<name>/config/stages/evaluate/graphrag.yaml
```

`EvidenceGraph` stages are planned only when training or running R-GCN-related paths needs them. GraphRAG runs directly from text candidates.

## Verification

```powershell
uv run pytest -q
uv run ruff check .
uv run basedpyright --level error
uv run python -m compileall -q graph_memory scripts tests
openspec validate adapt-traject-bench-dataset --strict
git diff --check
```

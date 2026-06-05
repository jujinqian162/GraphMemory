## 1. Loader and Stage Config Registry

- [x] 1.1 Add parser contract tests for retrieval, pair-build, train, evaluate, and experiment scripts
- [x] 1.2 Add `graph_memory.config` with `ConfigLoader.load(spec, argv)`, JSON codec, converter, patch merging, unknown-field rejection, and resolved config serialization
- [x] 1.3 Add `StageConfigSpec`, `StageId`, registry app root, and root config specs for prepare, graphs, pairs, tune, train, retrieve, evaluate, aggregate, experiment init, and experiment plan
- [x] 1.4 Add focused config loader and stage registry tests, including CLI-last precedence and fixed `profiles` / `default_profile` conventions

## 2. Retrieve Stage Registry Dispatch

- [x] 2.1 Add method-specific retrieval settings union and settings-type builder registry for BM25, dense, graph-rerank, and checkpoint-backed graph retrieval
- [x] 2.2 Add `graph_memory.stages.retrieve.run_retrieve_stage()` and migrate `scripts/run_retrieval.py` to load `Registry.configs.RETRIEVE`
- [x] 2.3 Move public retrieval method metadata source-of-truth into `graph_memory.registry` and downgrade `retrieval/catalog.py` plus `retrieval_registry.py` to compatibility projections
- [x] 2.4 Remove or contain old `builder_id` / public method string runtime dispatch from retrieval resolver/factory/application paths after registry dispatch covers those paths
- [x] 2.5 Run focused retrieval registry, run-retrieval, smoke retrieval, type, and architecture checks for the retrieve stage slice

## 3. Pair, Train, and Evaluate Stage Configs

- [x] 3.1 Add typed `PairBuildStageConfig` and migrate `scripts/build_train_pairs.py` to `Registry.configs.PAIRS`
- [x] 3.2 Add focused tests proving pair-build direct CLI overrides beat file config and hard dense encoder settings live inside pair-build job settings
- [x] 3.3 Add typed `TrainStageConfig`, R-GCN method/trainer/settings records, and a training builder registry entry
- [x] 3.4 Migrate `scripts/train_graph_retriever.py` and train stage orchestration to `Registry.training.build(config.job, deps)` without method-specific branches in the stage runner
- [x] 3.5 Add typed `EvaluateStageConfig` and migrate `scripts/evaluate_retrieval.py` without depending on retrieval method internals
- [x] 3.6 Run pair, training, evaluation, and experiment runner focused tests for the stage config migration

## 4. Typed Workflow Projection and Ablations

- [x] 4.1 Update workflow manifest generation to store/read resolved typed stage config projections while keeping existing manifest JSON readable
- [x] 4.2 Move R-GCN ablation variant patches into `graph_memory.registry.ablations`
- [x] 4.3 Downgrade `scripts/workflow/registry.py` to a workflow projection that does not independently own method or ablation semantics
- [x] 4.4 Run workflow orchestration and experiment runner focused tests

## 5. Config Schema and Legacy Cleanup

- [x] 5.1 Add schema v2 method config support with shallow `configs/methods/dense_rgcn_graph_retriever.json`
- [x] 5.2 Keep `configs/training/dense_rgcn_graph_retriever/base.json` readable through a compatibility adapter
- [x] 5.3 Remove production calls to old training config dict-slicing helpers
- [x] 5.4 Add architecture tests bounding method string dispatch, `builder_id`, compatibility projections, and old training helper calls
- [x] 5.5 Run the full validation gate: full pytest, basedpyright error-level, ruff, strict OpenSpec validation, and `git diff --check`

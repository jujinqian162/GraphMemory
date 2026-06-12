## 1. Dense Text Formatting Foundation

- [x] 1.1 Add failing tests proving dense inference and dense-ft data construction use identical query/passage text.
- [x] 1.2 Add `format_dense_query()` and `format_dense_passage()` to `graph_memory/embeddings/dense.py`.
- [x] 1.3 Refactor `DenseEncodingService._texts_for_request()` to use the shared helpers.
- [x] 1.4 Run `uv run pytest tests/test_batched_dense_encoding.py tests/test_dense_finetune_data.py -q`.

## 2. Dense Fine-Tune Data Builders

- [x] 2.1 Add tests for positive-only rows, hard-negative selection, task-qualified corpus ids, and unknown node-id errors.
- [x] 2.2 Create `graph_memory/models/dense_finetune` contracts for examples, build results, and data settings.
- [x] 2.3 Implement `build_dense_finetune_examples()`.
- [x] 2.4 Implement `build_ir_evaluator_payload()`.
- [x] 2.5 Run `uv run pytest tests/test_dense_finetune_data.py -q`.

## 3. Dense Fine-Tune Training Package

- [x] 3.1 Declare the SentenceTransformers runtime dependency in `pyproject.toml`.
- [x] 3.2 Run `uv lock` to update `uv.lock`.
- [x] 3.3 Add fake-model/fake-trainer tests for dense-ft training results, metadata, and metrics.
- [x] 3.4 Implement dense-ft training package skeleton without loading a real model in tests.
- [x] 3.5 Run `uv run pytest tests/test_dense_finetune_training.py -q`.

## 4. Method-Specific Train Stage

- [x] 4.1 Add tests that `Registry.configs.TRAIN` parses both R-GCN and dense-ft train configs.
- [x] 4.2 Add tests that train job settings dispatch to R-GCN or dense-ft by method id.
- [x] 4.3 Add `DenseFinetuneMethodSettings` and nested data/trainer/selection settings.
- [x] 4.4 Convert `TrainStageConfig` to a root-level method-specific union.
- [x] 4.5 Refactor `TrainingRegistry.build()` so it does not require global R-GCN dependencies for every method.
- [x] 4.6 Move R-GCN provider construction into the R-GCN-specific trainer/builder path.
- [x] 4.7 Run `uv run pytest tests/test_registry_stage_configs.py tests/test_dense_finetune_training.py -q`.

## 5. Real SentenceTransformers Training

- [x] 5.1 Build SentenceTransformers 2.7.0 `InputExample` rows and a PyTorch `DataLoader`.
- [x] 5.2 Load the base encoder with `SentenceTransformer(config.encoder.model_name)`.
- [x] 5.3 Map dense-ft trainer `device` settings to SentenceTransformers CPU/GPU behavior.
- [x] 5.4 Use `MultipleNegativesRankingLoss(model)`.
- [x] 5.5 Use `InformationRetrievalEvaluator` for dev ranking metrics.
- [x] 5.6 Pass dense-ft trainer settings directly to `SentenceTransformer.fit()`.
- [x] 5.7 Save the selected model directory and `dense_ft_model_config.json`.
- [x] 5.8 Run `uv run pytest tests/test_dense_finetune_training.py -q`.

## 6. Dense-FT Retrieval Registry

- [x] 6.1 Add `RetrievalMethodId.DENSE_FT = "dense_ft"`.
- [x] 6.2 Add `DenseFinetunedRetrievalSettings` with `top_k`, `checkpoint`, and `device`.
- [x] 6.3 Register dense-ft method metadata with checkpoint/model-directory and dense-encoder requirements.
- [x] 6.4 Add a retrieval builder that reads dense-ft metadata and returns `DenseTaskRetriever`.
- [x] 6.5 Run `uv run pytest tests/test_dense_ft_retrieval_registry.py tests/test_retrieval_registry_builders.py -q`.

## 7. Unified Train Script

- [x] 7.1 Create `scripts/train_method.py` using `CONFIG_LOADER.load(Registry.configs.TRAIN, argv)`.
- [x] 7.2 Load train artifacts according to the resolved stage config IO type.
- [x] 7.3 Call `run_train_stage(config, payload=...)`.
- [x] 7.4 Write metrics and run summary consistently for all train methods.
- [x] 7.5 Delete `scripts/train_graph_retriever.py` and update tests, workflow commands, and docs references.
- [x] 7.6 Change R-GCN train commands to `scripts/train_method.py --method dense_rgcn_graph_retriever`.
- [x] 7.7 Add dense-ft train commands with `scripts/train_method.py --method dense_ft`.
- [x] 7.8 Run `uv run pytest tests/test_phase2_rgcn_training.py tests/test_dense_finetune_training.py tests/test_cli_contracts.py -q`.

## 8. Workflow, Manifest, and Artifact Paths

- [x] 8.1 Add `WorkflowId.DENSE_FINETUNE_RETRIEVAL`.
- [x] 8.2 Add `DENSE_FT_WORKFLOW` with prepare, graphs, pairs, train, retrieve, evaluate, and aggregate stages.
- [x] 8.3 Register dense-ft in the method workflow registry.
- [x] 8.4 Use `learned/dense_ft/checkpoints/best_model` as dense-ft checkpoint/model-directory artifact.
- [x] 8.5 Generate all train commands through `scripts/train_method.py --method <method>`.
- [x] 8.6 Generate dense-ft retrieve commands with `--checkpoint <model_dir>` and without repeating encoder model CLI args.
- [x] 8.7 Generate dense-ft train/retrieve stage config projections.
- [x] 8.8 Run `uv run pytest tests/test_dense_ft_workflow.py tests/test_workflow_orchestration.py tests/test_experiment_runner.py -q`.

## 9. Experiment Config and Docs

- [x] 9.1 Add `dense_ft` to `configs/experiments/hotpotqa_evidence_retrieval.json` methods.
- [x] 9.2 Add dense-ft training config mapping to the experiment config.
- [x] 9.3 Create `configs/training/dense_ft/base.json` with override-only profiles.
- [x] 9.4 Create `docs/configs/training/dense_ft/base.md`.
- [x] 9.5 Update `docs/40-operations/commands.md` with dense-ft smoke, quick, and full commands.
- [x] 9.6 Run `uv run pytest tests/test_experiment_runner.py -q`.

## 10. Verification

- [x] 10.1 Run focused dense-ft tests: `uv run pytest tests/test_dense_finetune_data.py tests/test_dense_finetune_training.py tests/test_dense_ft_retrieval_registry.py tests/test_dense_ft_workflow.py -q`.
- [x] 10.2 Run dense/R-GCN regression tests: `uv run pytest tests/test_batched_dense_encoding.py tests/test_registry_stage_configs.py tests/test_retrieval_registry_builders.py tests/test_experiment_runner.py tests/test_workflow_orchestration.py tests/test_phase2_rgcn_training.py -q`.
- [x] 10.3 Run `uv run basedpyright graph_memory scripts tests --level error`.
- [x] 10.4 Run `uv run python scripts/experiment.py methods list`.
- [x] 10.5 Run `uv run python scripts/experiment.py plan dense_ft_smoke --profile smoke --methods dense_ft --force`.
- [x] 10.6 Run strict OpenSpec validation for `add-dense-ft-retrieval`.
- [x] 10.7 Add a stage-level dense-ft retrieval regression test and route `DenseFinetunedRetrievalSettings` through `FlatRetrievalBuildPayload`.

## 11. SentenceTransformers 2.7.0 Unified Backend

- [x] 11.1 Replace Trainer-based tests with failing tests for `InputExample`, `DataLoader`, evaluator, and `SentenceTransformer.fit()`.
- [x] 11.2 Replace dense-ft Trainer integration with a single SentenceTransformers 2.7.0 `fit()` implementation.
- [x] 11.3 Remove Trainer-only dense-ft settings and expose only 2.7.0 fit settings.
- [x] 11.4 Pin `sentence-transformers==2.7.0`, remove direct `datasets` and `accelerate` dependencies, and update `uv.lock`.
- [x] 11.5 Update dense-ft config, CLI contracts, implementation plan, and configuration docs.
- [x] 11.6 Run focused dense-ft tests and Python 3.10 compatibility checks.
- [x] 11.7 Run the full test suite, BasedPyright, experiment planning smoke, and strict OpenSpec validation.

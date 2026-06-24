# Dense-FT Seeded R-GCN Method Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `dense_ft_rgcn_graph_retriever` as a separate public experiment method whose R-GCN training uses the workflow-produced `dense_ft` model directory as its seed/text encoder.

**Architecture:** `dense_ft_rgcn_graph_retriever` is a new method id and result-table row, but it reuses the existing Dense-FT trainer, R-GCN trainer, R-GCN checkpoint format, and trainable graph retrieval implementation. The workflow owns the dependency edge: when this method is selected, it trains or reuses `dense_ft`, then passes `learned/dense_ft/checkpoints/best_model` into the R-GCN train stage as an input artifact.

**Tech Stack:** Python dataclasses, existing `Registry` method definitions, current workflow manifest/planner/stage-config compiler, `scripts/train_method.py`, Dense-FT SentenceTransformer model directories, R-GCN `.pt` checkpoints, pytest, basedpyright, `uv`.

---

Date: 2026-06-24

Status: Draft implementation plan.

## 1. Decision Summary

Implement a new public method:

```text
dense_ft_rgcn_graph_retriever
```

Do not implement a second graph retriever. Do not expose this as a user-selectable encoder option inside `dense_rgcn_graph_retriever`. The public method identity is the experiment contract:

```text
dense_rgcn_graph_retriever
  seed/text encoder: configured base dense encoder
  train artifact: learned/dense_rgcn_graph_retriever/checkpoints/best.pt

dense_ft_rgcn_graph_retriever
  upstream train dependency: dense_ft
  seed/text encoder: learned/dense_ft/checkpoints/best_model
  train artifact: learned/dense_ft_rgcn_graph_retriever/checkpoints/best.pt
```

The key implementation boundary is:

```text
experiment method id
  -> workflow dependency graph
  -> stage config input artifact
  -> existing R-GCN trainer
  -> existing R-GCN checkpoint-backed retriever
```

## 2. Non-Goals

- Do not add a new R-GCN model class.
- Do not duplicate `TrainableGraphRetrievalMethod`.
- Do not duplicate Dense-FT retrieval or dense scoring logic.
- Do not make users hand-write a dense-ft model path in experiment configs.
- Do not add a general plugin/dependency framework beyond this train dependency shape.
- Do not compute path-metric capability from method-name string heuristics.

## 3. Target Call Flow

Selecting only `dense_ft_rgcn_graph_retriever` should produce this execution graph:

```text
prepare
  |
graphs
  |
pairs: dense_ft
  |
pairs: dense_ft_rgcn_graph_retriever
  |
train: dense_ft
  -> learned/dense_ft/checkpoints/best_model
       |
       v
train: dense_ft_rgcn_graph_retriever
  -> learned/dense_ft_rgcn_graph_retriever/checkpoints/best.pt
       |
       v
retrieve: dense_ft_rgcn_graph_retriever
  |
evaluate: dense_ft_rgcn_graph_retriever
  |
aggregate
```

If the user also selects `dense_ft`, the same `dense_ft` artifacts are reused and `dense_ft` additionally gets retrieve/evaluate/table output. If the user does not select `dense_ft`, it remains a dependency-only train unit and does not create prediction/metric rows.

## 4. File Responsibility Map

### Registry and Type Surface

- Modify `graph_memory/registry/retrieval.py`
  - Add `RetrievalMethodId.DENSE_FT_RGCN_GRAPH_RETRIEVER`.
  - Widen `CheckpointGraphRetrievalSettings.method` to accept both R-GCN method ids.

- Modify `graph_memory/registry/methods.py`
  - Register `dense_ft_rgcn_graph_retriever` as R-GCN-trainable, graph-backed, checkpoint-backed, path-metric-capable.
  - Add explicit train dependency metadata, not a config option.
  - Set `seed_method=RetrievalMethodId.DENSE_FT` for the new method.

- Modify `graph_memory/registry/method_configs.py`
  - Reuse `RgcnMethodConfig`, `RgcnMethodSettings`, and `RgcnTrainSettings`.
  - Widen method literals to include both `dense_rgcn_graph_retriever` and `dense_ft_rgcn_graph_retriever`.

- Modify `graph_memory/registry/stage_configs.py`
  - Widen `RgcnTrainStageConfig.method`.
  - Add optional `seed_checkpoint: Path | None` to `RgcnTrainIO`.

### R-GCN Training and Checkpoint Metadata

- Modify `graph_memory/models/graph_retriever/config/defaults.py`
  - Replace hard-coded `method_name="dense_rgcn_graph_retriever"` with a required `method_name` argument.

- Modify `graph_memory/stages/train_payloads.py`
  - Add `seed_checkpoint: Path | None = None` to `RgcnTrainPayload`.

- Modify `graph_memory/stages/trainers.py`
  - Let `RgcnGraphRetrieverTrainer` resolve the actual encoder settings from `payload.seed_checkpoint` when present.
  - For dense-ft seeded runs, load `dense_ft_model_config.json` from the model directory and build `DenseGraphFeatureProvider(model_name=str(seed_checkpoint), query_prefix=metadata.query_prefix, passage_prefix=metadata.passage_prefix, batch_size=metadata.batch_size)`.
  - Save R-GCN checkpoint metadata with `method_name=config.method.value` and `encoder_model=str(seed_checkpoint)` for dense-ft seeded runs.

- Modify `scripts/train_method.py`
  - Read `config.io.seed_checkpoint` into `RgcnTrainPayload`.
  - Include `seed_checkpoint` in train run-summary inputs when present.

### Workflow Dependency Management

- Modify `scripts/workflow/manifest.py`
  - Keep `selected_methods` as public output methods.
  - Add dependency-only learned artifacts for train dependencies such as `dense_ft`.
  - Write stage configs for dependency train methods even when they are not selected public output methods.
  - Do not create predictions, metrics, or failure-case paths for dependency-only methods unless selected.

- Modify `scripts/workflow/planner.py`
  - Expand train dependencies when planning `pairs` and `train`.
  - Topologically order train commands so `dense_ft` runs before `dense_ft_rgcn_graph_retriever`.
  - Validate `--from train` and `--from retrieve` fail early when required dependency artifacts are missing.

- Modify `scripts/workflow/stage_configs.py`
  - When building `dense_ft_rgcn_graph_retriever` train config, set `RgcnTrainIO.seed_checkpoint` to `manifest["artifacts"]["learned"]["dense_ft"]["best_checkpoint"]`.
  - Keep ordinary `dense_rgcn_graph_retriever` train config with `seed_checkpoint=None`.
  - Build retrieval settings for both R-GCN methods through `CheckpointGraphRetrievalSettings`.

- Modify `scripts/workflow/workflows.py` and `scripts/workflow/registry.py`
  - Reuse `RGCN_WORKFLOW` for the new method lifecycle.
  - Do not add a separate workflow unless command ordering cannot be kept in planner metadata.

### Method Configs, Experiment Configs, and Docs

- Create `configs/methods/dense_ft_rgcn_graph_retriever.json`
  - Copy current R-GCN train/model/pair defaults.
  - Change only `"method"` to `"dense_ft_rgcn_graph_retriever"` unless tests prove another field is required.

- Modify experiment configs that currently list both `dense_ft` and `dense_rgcn_graph_retriever`
  - Add `dense_ft_rgcn_graph_retriever` to method lists where the default trainable comparison should include it.
  - Add method config mapping to `configs/methods/dense_ft_rgcn_graph_retriever.json`.
  - Expected active files include:
    - `configs/experiments/hotpotqa_evidence_retrieval.json`
    - `configs/experiments/hotpoqa_dev_full.json`
    - `configs/experiments/2wiki_evidence_retrieval.json`
    - `configs/experiments/2wiki_tiny.json`

- Modify active docs/tests with hard-coded method lists
  - `README.md`
  - `docs/40-operations/commands.md`
  - config docs under `docs/configs/experiments/` if present for touched configs.

## 5. Implementation Tasks

### Task 1: Add the Public Method Surface

**Files:**
- Modify: `graph_memory/registry/retrieval.py`
- Modify: `graph_memory/registry/methods.py`
- Modify: `graph_memory/registry/method_configs.py`
- Modify: `graph_memory/registry/stage_configs.py`
- Test: `tests/test_method_registry.py`
- Test: `tests/test_registry_stage_configs.py`

- [ ] Add `DENSE_FT_RGCN_GRAPH_RETRIEVER = "dense_ft_rgcn_graph_retriever"` to `RetrievalMethodId`.
- [ ] Widen R-GCN related method literals so existing R-GCN config/settings/stage types accept both method ids.
- [ ] Add method definition with:

```python
MethodDefinition(
    identifier=RetrievalMethodId.DENSE_FT_RGCN_GRAPH_RETRIEVER,
    lifecycle=RetrievalLifecycle.RGCN_TRAINABLE,
    retrieval_settings_type=CheckpointGraphRetrievalSettings,
    dependencies=RetrievalDependencySpec(
        graphs=GraphInputSource.GRAPH_ARTIFACT,
        selected_config=SelectedConfigSource.NONE,
        model=ModelSource.CHECKPOINT_FILE,
        encoder=EncoderSource.CHECKPOINT_METADATA,
    ),
    method_config_type=RgcnMethodConfig,
    train_artifact=TrainArtifactSpec("best.pt", ArtifactKind.FILE),
    seed_method=RetrievalMethodId.DENSE_FT,
    train_dependencies=(RetrievalMethodId.DENSE_FT,),
)
```

- [ ] If `MethodDefinition` does not yet have `train_dependencies`, add it as `tuple[RetrievalMethodId, ...] = ()`.
- [ ] Add a registry test proving the new method:
  - is listed by `Registry.methods.list_ids()`;
  - has lifecycle `RGCN_TRAINABLE`;
  - supports path metrics through existing registry capability logic;
  - declares `seed_method == RetrievalMethodId.DENSE_FT`;
  - declares train dependency `(RetrievalMethodId.DENSE_FT,)`.
- [ ] Add a stage-config test proving `RgcnTrainStageConfig` accepts `method=RetrievalMethodId.DENSE_FT_RGCN_GRAPH_RETRIEVER`.

### Task 2: Make R-GCN Checkpoints Preserve the Selected Method Identity

**Files:**
- Modify: `graph_memory/models/graph_retriever/config/defaults.py`
- Modify: `graph_memory/stages/trainers.py`
- Modify: `scripts/train_method.py`
- Test: `tests/test_phase2_rgcn_training.py`

- [ ] Change `default_model_config()` to require `method_name: str`.
- [ ] Pass `method_name=self.settings.method.value` from `RgcnGraphRetrieverTrainer`.
- [ ] Add a training unit test with a fake R-GCN payload verifying a `dense_ft_rgcn_graph_retriever` training result has `result.model_config.method_name == "dense_ft_rgcn_graph_retriever"`.
- [ ] Add a checkpoint save/load test verifying `load_rgcn_checkpoint(..., expected_method="dense_ft_rgcn_graph_retriever")` accepts the new method and rejects the old one for that checkpoint.

### Task 3: Pass Dense-FT Model Directory into R-GCN Training

**Files:**
- Modify: `graph_memory/registry/stage_configs.py`
- Modify: `graph_memory/stages/train_payloads.py`
- Modify: `graph_memory/stages/trainers.py`
- Modify: `scripts/train_method.py`
- Test: `tests/test_phase2_rgcn_training.py`
- Test: `tests/test_current_trainable_artifacts.py`

- [ ] Add `seed_checkpoint: Path | None = None` to `RgcnTrainIO`.
- [ ] Add `seed_checkpoint: Path | None = None` to `RgcnTrainPayload`.
- [ ] Update `scripts/train_method.py` so `_load_payload()` copies `config.io.seed_checkpoint` into the payload.
- [ ] Update `_input_paths()` so run summaries include `"seed_checkpoint": "<path>"` only when present.
- [ ] Add a helper in `graph_memory/stages/trainers.py`:

```python
def _effective_rgcn_encoder_settings(settings: RgcnMethodSettings, seed_checkpoint: Path | None) -> DenseEncoderSettings:
    if seed_checkpoint is None:
        return settings.encoder
    metadata = load_dense_ft_model_metadata(seed_checkpoint)
    return DenseEncoderSettings(
        model_name=str(seed_checkpoint),
        query_prefix=metadata.query_prefix,
        passage_prefix=metadata.passage_prefix,
        batch_size=metadata.batch_size,
    )
```

- [ ] Use the effective encoder settings for:
  - `DenseGraphFeatureProvider`;
  - `default_model_config()`;
  - checkpoint metadata.
- [ ] Add a unit test that writes a minimal dense-ft model metadata file under a fake `best_model` directory and verifies the R-GCN trainer uses that path/prefix/batch in `model_config`.

### Task 4: Compile Workflow Dependency Artifacts and Stage Configs

**Files:**
- Modify: `scripts/workflow/manifest.py`
- Modify: `scripts/workflow/stage_configs.py`
- Test: `tests/test_current_manifest_contract.py`
- Test: `tests/test_dense_ft_workflow.py`
- Test: `tests/test_experiment_runner.py`

- [ ] Add a dependency expansion helper that derives train dependencies from `Registry.methods.get(method).train_dependencies`.
- [ ] Keep `manifest["selected_methods"]` unchanged as the user-selected public methods.
- [ ] Add dependency-only learned artifacts for `dense_ft` when only `dense_ft_rgcn_graph_retriever` is selected.
- [ ] Write `pairs` and `train` stage configs for dependency-only methods.
- [ ] Do not write dependency-only `retrieve` or `evaluate` stage configs.
- [ ] In `_train_stage_config()` for the new method, set:

```python
seed_checkpoint=Path(manifest["artifacts"]["learned"]["dense_ft"]["best_checkpoint"])
```

- [ ] Add a manifest test:
  - initialize with `--methods dense_ft_rgcn_graph_retriever`;
  - assert `artifacts.learned.dense_ft.best_checkpoint` exists in the manifest;
  - assert `stage_configs.train.dense_ft` exists;
  - assert `stage_configs.train.dense_ft_rgcn_graph_retriever` exists;
  - assert no `predictions.dense_ft` exists unless `dense_ft` is selected.
- [ ] Add a stage-config test proving the new method's train config contains `io.seed_checkpoint == learned/dense_ft/checkpoints/best_model`.

### Task 5: Order Planner Commands and Validate Missing Dependencies

**Files:**
- Modify: `scripts/workflow/planner.py`
- Test: `tests/test_experiment_runner.py`
- Test: `tests/test_workflow_resume.py` or the current planner/resume test file if differently named.

- [ ] Expand methods for `pairs` and `train` planning to include dependency-only train methods.
- [ ] Ensure train commands are ordered:

```text
train dense_ft
train dense_ft_rgcn_graph_retriever
```

- [ ] Ensure retrieve/evaluate/aggregate commands still use only public selected methods.
- [ ] Add a test for plan rendering with only `dense_ft_rgcn_graph_retriever` selected. Expected rendered command order:

```text
scripts/build_train_pairs.py ... dense_ft ...
scripts/build_train_pairs.py ... dense_ft_rgcn_graph_retriever ...
scripts/train_method.py ... dense_ft ...
scripts/train_method.py ... dense_ft_rgcn_graph_retriever ...
scripts/run_retrieval.py ... dense_ft_rgcn_graph_retriever ...
scripts/evaluate_retrieval.py ... dense_ft_rgcn_graph_retriever ...
scripts/aggregate_tables.py ...
```

- [ ] Add fail-fast validation:
  - `--from train` without existing dense-ft train pairs fails before R-GCN training;
  - `--from retrieve` without `dense_ft_rgcn_graph_retriever` best checkpoint fails;
  - R-GCN train without `dense_ft` best model fails with a message naming the missing dense-ft model path.

### Task 6: Reuse Checkpoint-Backed Retrieval for Both R-GCN Methods

**Files:**
- Modify: `graph_memory/registry/retrieval.py`
- Modify: `graph_memory/registry/retrieval_builders.py`
- Modify: `scripts/workflow/stage_configs.py`
- Test: `tests/test_phase2_rgcn_retrieval.py`
- Test: `tests/test_retrieval_provenance.py`

- [ ] Widen `CheckpointGraphRetrievalSettings.method` to both R-GCN ids.
- [ ] Keep `_build_checkpoint_graph()` as the only checkpoint-backed graph retriever builder.
- [ ] Verify retrieval loads the checkpoint with `expected_method=settings.method.value`.
- [ ] Add a retrieval registry test that builds settings for `dense_ft_rgcn_graph_retriever` and confirms:
  - checkpoint path is `learned/dense_ft_rgcn_graph_retriever/checkpoints/best.pt`;
  - method provenance is `dense_ft_rgcn_graph_retriever`;
  - encoder provenance comes from checkpoint metadata, not experiment config.

### Task 7: Add Method Configs and Public Experiment Wiring

**Files:**
- Create: `configs/methods/dense_ft_rgcn_graph_retriever.json`
- Modify: `configs/experiments/hotpotqa_evidence_retrieval.json`
- Modify: `configs/experiments/hotpoqa_dev_full.json`
- Modify: `configs/experiments/2wiki_evidence_retrieval.json`
- Modify: `configs/experiments/2wiki_tiny.json`
- Modify: `README.md`
- Modify: `docs/40-operations/commands.md`
- Test: `tests/test_twowiki_workflow.py`
- Test: `tests/test_cli_contracts.py`

- [ ] Copy current `configs/methods/dense_rgcn_graph_retriever.json` to `configs/methods/dense_ft_rgcn_graph_retriever.json`.
- [ ] Change only `"method"` in the copied method config unless a failing test proves another field is necessary.
- [ ] Add method config mappings in experiment configs.
- [ ] Add the method to default method lists where trainable baseline comparisons are maintained.
- [ ] Update docs and hard-coded test expectations in the same patch so visible method lists do not drift.

### Task 8: Verification

**Files:**
- No new production files unless failures identify missing coverage.

- [ ] Run targeted tests:

```powershell
uv run pytest tests/test_method_registry.py tests/test_registry_stage_configs.py tests/test_current_manifest_contract.py tests/test_experiment_runner.py tests/test_phase2_rgcn_training.py tests/test_phase2_rgcn_retrieval.py tests/test_dense_ft_workflow.py -q
```

- [ ] Run type check:

```powershell
uv run basedpyright --level error
```

- [ ] Run OpenSpec validation if an OpenSpec change is created for this implementation:

```powershell
openspec validate <change-name> --strict
```

- [ ] Run a named workflow plan check:

```powershell
uv run python scripts/experiment.py plan dense_ft_rgcn_smoke --config configs/experiments/2wiki_tiny.json --profile smoke --methods dense_ft_rgcn_graph_retriever --force
```

- [ ] On a CUDA-capable environment, run a smoke workflow:

```powershell
uv run python scripts/experiment.py run dense_ft_rgcn_smoke --config configs/experiments/2wiki_tiny.json --profile smoke --methods dense_ft_rgcn_graph_retriever --force
```

- [ ] Verify the run produces:
  - `learned/dense_ft/checkpoints/best_model/dense_ft_model_config.json`;
  - `learned/dense_ft_rgcn_graph_retriever/checkpoints/best.pt`;
  - `predictions/test.dense_ft_rgcn_graph_retriever.ranked.json`;
  - `metrics/test.dense_ft_rgcn_graph_retriever.metrics.csv`;
  - numeric path metrics for `dense_ft_rgcn_graph_retriever`.

## 6. Acceptance Criteria

- `scripts/experiment.py methods list` shows `dense_ft_rgcn_graph_retriever` as a separate public method.
- Selecting only `dense_ft_rgcn_graph_retriever` schedules dense-ft training as a dependency.
- Selecting only `dense_ft_rgcn_graph_retriever` does not emit a `dense_ft` prediction or metric row.
- Selecting both `dense_ft` and `dense_ft_rgcn_graph_retriever` reuses the same dense-ft learned artifact.
- R-GCN checkpoint metadata for the new method stores:
  - `method_name = dense_ft_rgcn_graph_retriever`;
  - `encoder_model = <dense_ft best_model path>`;
  - dense-ft query/passage prefixes and batch size from dense-ft metadata.
- R-GCN retrieval for the new method uses the existing checkpoint-backed graph retriever path.
- `dense_rgcn_graph_retriever` behavior remains unchanged.
- Path metrics remain registry-capability gated and numeric for both R-GCN methods.

## 7. Risks

- Command ordering is the main risk. The current workflow stage order groups all train commands together, so dependency ordering inside the train stage must be explicit.
- Manifest shape can drift if dependency-only methods are mixed into `selected_methods`. Keep public output methods and dependency execution methods separate.
- R-GCN checkpoint validation currently expects exact method identity. Update method literals and `default_model_config()` first to avoid confusing checkpoint failures.
- Dense-ft metadata must own prefix/batch semantics for dense-ft seeded R-GCN. Do not infer these values from the base R-GCN config.


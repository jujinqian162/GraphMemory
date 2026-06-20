# Workflow-Driven Ablation Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add config-controlled R-GCN ablation execution while moving experiment lifecycle decisions into typed workflow adapters under `scripts/workflow/`.

**Architecture:** Keep `scripts/experiment.py` as the user entry point and low-level scripts as explicit IO adapters. Add a typed workflow registry, deterministic artifact namespaces, declarative invalidation dimensions, strict resume validation, and a thin `graph_memory.experiment` compatibility facade.

**Tech Stack:** Python 3.12, `enum.StrEnum`, dataclasses, argparse, pytest, OpenSpec.

---

## File Map

Create:

- `scripts/workflow/__init__.py`: public orchestration exports.
- `scripts/workflow/types.py`: closed enums and typed planning records.
- `scripts/workflow/workflows.py`: stateless, graph-rerank, and R-GCN lifecycle adapters.
- `scripts/workflow/registry.py`: static method-to-workflow and suite registrations.
- `scripts/workflow/artifacts.py`: main and variant paths plus alias records.
- `scripts/workflow/manifest.py`: config resolution, manifest initialization, and schema compatibility.
- `scripts/workflow/planner.py`: stage selection, run-unit expansion, invalidation, dependency checks, command rendering, and execution.
- `scripts/workflow/status.py`: ordinary and variant status inspection.
- `tests/test_workflow_orchestration.py`: focused workflow, alias, resume, config, and CLI discovery tests.

Modify:

- `graph_memory/experiment.py`: compatibility re-export facade.
- `scripts/experiment.py`: workflow imports and ablation CLI controls.
- `graph_memory/learned/training.py`: edge-view model variants.
- `graph_memory/learned/inference.py`: model-visible prediction subgraphs.
- `scripts/aggregate_tables.py`: optional indexed ablation aggregation.
- `tests/test_experiment_runner.py`: non-ablation regression and ablation integration coverage.
- `tests/test_phase2_rgcn_training.py`: model config edge-view coverage.
- `tests/test_phase2_rgcn_retrieval.py`: visible-edge inference coverage.
- `tests/test_phase1_real_cli_smoke.py`: indexed aggregation coverage.
- `configs/experiments/*.json`: explicit disabled defaults and selected-variant example.
- `configs/training/dense_rgcn_graph_retriever/ablations.json`: selection metadata only.
- `docs/configs/` and `docs/40-operations/`: operation and extension docs.

## Batch 1: Typed Workflow Contracts

- [ ] Write tests importing `StageId`, `WorkflowId`, `ArtifactRole`, `ChangeDimension`, `ArtifactState`, and `RgcnAblationVariant`.
- [ ] Run `pytest tests/test_workflow_orchestration.py -q --basetemp .pytest_tmp -p no:cacheprovider` and confirm import failure.
- [ ] Add enum and dataclass definitions, ordered workflow adapters, registry validation, and discovery helpers.
- [ ] Re-run the focused test and confirm pass.

Expected closed values include:

```python
class ChangeDimension(StrEnum):
    PAIR_SAMPLING = "pair_sampling"
    MODEL_STRUCTURE = "model_structure"
    MODEL_GRAPH_VIEW = "model_graph_view"
```

The R-GCN steps declare:

```python
WorkflowStepSpec(stage=StageId.PAIRS, invalidated_by=frozenset({ChangeDimension.PAIR_SAMPLING}))
WorkflowStepSpec(
    stage=StageId.TRAIN,
    invalidated_by=frozenset({ChangeDimension.PAIR_SAMPLING, ChangeDimension.MODEL_STRUCTURE, ChangeDimension.MODEL_GRAPH_VIEW}),
)
```

## Batch 2: Artifact And Planner Migration

- [ ] Add tests proving main paths remain stable when ablation is disabled.
- [ ] Add tests proving `wo_graph` aliases main pairs and starts local allocation at `train`.
- [ ] Add tests proving `wo_hard_negatives` owns local pairs and downstream artifacts.
- [ ] Add tests proving `full_rgcn` aliases main artifacts and schedules no duplicate commands.
- [ ] Add tests proving `--from retrieve` succeeds only with selected variant checkpoints.
- [ ] Move manifest, artifact, planner, and status logic under `scripts/workflow/`.
- [ ] Keep `graph_memory.experiment` as a compatibility facade and run runner regression tests.

Use a planner API shaped as:

```python
build_stage_plan(
    manifest,
    methods=None,
    from_stage=None,
    to_stage=None,
    variants=None,
    ablations_only=False,
)
```

## Batch 3: Config And CLI Integration

- [ ] Add tests for omitted or false `enable_ablation`.
- [ ] Add tests for configured ordered subsets, unknown variants, discovery, repeated `--variant`, and `--ablations-only`.
- [ ] Layer minimal variant overrides over the resolved main training config and persist generated configs.
- [ ] Add `scripts/experiment.py ablations list [--method ...]`.
- [ ] Update configs and retire the duplicated hand-maintained ablation training template.

The only user-maintained full R-GCN config remains:

```text
configs/training/dense_rgcn_graph_retriever/base.json
```

## Batch 4: Low-Level R-GCN Variant Support

- [ ] Add model config tests for `wo_entity_overlap`, `wo_sequential`, and `wo_query_overlap`.
- [ ] Add an inference test proving hidden edge types are absent from returned subgraph edges.
- [ ] Extend model config construction and reuse its enabled edge policy during inference.
- [ ] Add a pair-sampling inheritance test for `wo_hard_negatives`.

## Batch 5: Aggregation And Documentation

- [ ] Add an indexed ablation aggregation test with aliased `full_rgcn`.
- [ ] Extend `scripts/aggregate_tables.py` with optional `--ablation_index` and `--output_ablation`.
- [ ] Write the stable ablation result columns.
- [ ] Update config, CLI, resume, artifact-layout, and retriever-extension documentation.

## Batch 6: Verification

- [ ] Run focused workflow and R-GCN tests with `--basetemp .pytest_tmp`.
- [ ] Run the full test suite with `--basetemp .pytest_tmp`.
- [ ] Run `ruff check graph_memory scripts tests`.
- [ ] Run `git diff --check`.
- [ ] Run `openspec validate refactor-experiment-workflow-ablation --strict`.
- [ ] Review `git status --short` and confirm unrelated user files were not reverted.

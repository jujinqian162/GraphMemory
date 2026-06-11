# Implementation Handoff

Date: 2026-06-03

Status: Maintained implementation handoff.

## Purpose

This document explains how to read and review the implementation. `docs/40-operations/commands.md` explains how to run it.

## Review Entry Points

Recommended reading order:

1. `docs/00-overview/project-overview.md`
   Confirms the scientific boundary: HotpotQA evidence tracing, not answer generation.
2. `docs/20-contracts/data-contracts.md`
   Defines disk artifact schemas.
3. `docs/20-contracts/retrieval-contracts.md`
   Defines public method names, retrieval contracts, registry metadata, and seed signals.
4. `docs/20-contracts/model-contracts.md`
   Defines trainable model config, tensor batch, checkpoint, and training contracts.
5. `docs/30-design/architecture.md` and `docs/30-design/abstractions.md`
   Explain package ownership, dependency direction, and abstraction boundaries.
6. `scripts/experiment.py`, `scripts/workflow/`, and the five root workflow integration ports.
7. Low-level `scripts/*.py` CLI adapters.
8. Domain packages under `graph_memory/`.

## Main Control Flow

```text
scripts/prepare_hotpotqa.py
  -> graph_memory.datasets.splits.sample_split
  -> graph_memory.datasets.hotpotqa.parser.parse_hotpotqa_examples
  -> graph_memory.datasets.hotpotqa.converter.convert_hotpotqa_examples
  -> graph_memory.validation task validators

scripts/build_graphs.py
  -> graph_memory.graphs.construction.builder.build_graphs
  -> graph_memory.graphs.statistics.graph_statistics
  -> graph_memory.validation.validate_graphs

scripts/run_retrieval.py
  -> graph_memory.config.CONFIG_LOADER.load(Registry.configs.RETRIEVE, argv)
  -> graph_memory.stages.retrieve.run_retrieve_stage
  -> graph_memory.registry.retrieval_builders.RETRIEVAL_REGISTRY
  -> graph_memory.retrieval.execution.service.run_retrieval
  -> graph_memory.validation.validate_ranked_results

scripts/tune_graph_rerank.py
  -> graph_memory.retrieval.tuning.tune_graph_rerank
  -> graph_memory.retrieval.tuning.initial_scores
  -> graph_memory.retrieval.methods.graph_rerank
  -> graph_memory.evaluation.evaluate_results

scripts/build_train_pairs.py
  -> graph_memory.training_pairs.build_train_pairs
  -> graph_memory.validation.validate_train_pairs

scripts/train_method.py
  -> graph_memory.stages.train.run_train_stage
  -> graph_memory.registry.training
  -> method-specific trainer and artifact output

scripts/evaluate_retrieval.py
  -> graph_memory.evaluation.evaluate_results

scripts/aggregate_tables.py
  -> graph_memory.evaluation.tables

scripts/experiment.py
  -> scripts.workflow
  -> existing low-level scripts with explicit generated input/output paths
```

## Key Abstractions

| Abstraction | Location | What it does | Must not do | Tests |
|---|---|---|---|---|
| `MemoryTaskInput` | `graph_memory/contracts/tasks.py` | Input-visible query and memory sentence artifact shape. | Contain labels or answer text. | `tests/test_phase1_real_validation.py`, `tests/test_phase1_real_data_structures.py` |
| `MemoryTaskLabels` | `graph_memory/contracts/tasks.py` | Gold answer and evidence labels for evaluation/tuning. | Feed graph construction or retrieval. | `tests/test_phase1_real_validation.py` |
| `HotpotQAExample` / `HotpotQAConversionResult` | `graph_memory/datasets/hotpotqa/` | Typed raw HotpotQA parse result and named conversion output. | Expose raw `dict` records or tuple-packed outputs. | `tests/test_phase1_real_data_structures.py` |
| `MemoryGraph` | `graph_memory/contracts/graphs.py` | Typed graph over `q` and memory sentence nodes. | Read label-only fields. | `tests/test_phase1_real_graphs.py` |
| `GraphBuilder` | `graph_memory/graphs/construction/builder.py` | Applies ordered graph edge rules to input-visible task records. | Run retrieval or evaluation. | `tests/test_core_refactor_batch3_boundaries.py` |
| `RankedNode` | `graph_memory/retrieval/contracts.py` | Internal scored memory-node result. | Represent persisted JSON directly. | `tests/test_phase1_real_retrieval.py` |
| `RankedResult` | `graph_memory/contracts/ranking.py` | Persisted ranked-result artifact shape. | Drop unselected memory nodes. | `tests/test_phase1_real_retrieval.py` |
| `RetrieveStageConfig` | `graph_memory/registry/stage_configs.py` | Stage-level request for one complete retrieval run. | Carry unrelated method-family optional bags. | `tests/test_retrieval_domain_boundaries.py`, `tests/test_registry_stage_configs.py` |
| `Retriever` | `graph_memory/retrieval/contracts.py` | Single-task complete ranking protocol. | Compute metrics or read labels. | `tests/test_phase1_real_retrieval.py` |
| `RetrievalMethodSpec` catalog | `graph_memory/retrieval/catalog.py` through `graph_memory/retrieval_registry.py` | Single source for public method names and capabilities. | Import concrete retrieval builders or duplicate method lists elsewhere. | `tests/test_phase1_real_retrieval.py`, `tests/test_experiment_runner.py` |
| `RetrievalMethod` | `graph_memory/retrieval/contracts.py` | Internal boundary for a public method that emits final ranked nodes and retrieved edges. | Force every future baseline to be a weighted sum. | `tests/test_phase1_real_retrieval.py` |
| `FlatRetrievalMethod` | `graph_memory/retrieval/methods/flat/method.py` | Wraps BM25 and dense seed retrievers for flat public methods. | Own graph-rerank score composition, labels, metrics, or file IO. | `tests/test_phase1_real_retrieval.py` |
| `InitialScoreCache` | `graph_memory/retrieval/tuning/initial_scores.py` | Holds per-task seed scores for one tuning invocation. | Persist scores or become an artifact contract. | `tests/test_phase1_real_retrieval.py` |
| Graph-rerank components | `graph_memory/retrieval/methods/graph_rerank/` | Candidate expansion, score components, normalization, config, and method adapter. | Select experiment workflow stages or read labels. | `tests/test_phase1_real_retrieval.py` |
| `GraphRerankConfig` | `graph_memory/retrieval/methods/graph_rerank/config.py` | Graph score propagation config; rejects deprecated `type_weights`. | Treat `query_overlap` as a neighbor type weight. | `tests/test_phase1_real_retrieval.py` |
| Train-pair builder | `graph_memory/training_pairs/builder.py` | Produces deterministic positive/negative train-pair artifacts. | Depend on trainable model internals. | `tests/test_phase2_rgcn_pairs.py` |
| `GraphBatch` / `TrainingBatch` | `graph_memory/models/graph_retriever/internals/contracts.py` | Tensor batch contracts used by graph-scoring model code. | Leak raw artifact dictionaries into model forward. | `tests/test_phase2_rgcn_model.py`, `tests/test_phase2_rgcn_training.py` |
| Trainable graph retriever | `graph_memory/models/graph_retriever/` and `graph_memory/retrieval/methods/trainable_graph.py` | Checkpoint-backed graph model training and retrieval adapter. | Parse CLI args or read labels during inference. | `tests/test_phase2_rgcn_training.py`, `tests/test_phase2_rgcn_retrieval.py` |
| Validators | `graph_memory/validation/` | Enforce contracts from `object` boundaries and narrow loaded JSON/domain artifacts after runtime shape checks. | Repair, sort, drop, infer, or copy records just to satisfy IDE types. | `tests/test_phase1_real_validation.py` |
| Metric primitives | `graph_memory/evaluation/metrics.py` and `graph_memory/evaluation/connectivity.py` | Compute node and connectivity metrics. | Re-run retrieval or read task inputs for gold fields. | `tests/test_phase1_real_evaluation.py` |
| Run summaries | `graph_memory/infrastructure/run_summary.py` through `graph_memory/observability.py` | Preserve config, paths, counts, timings, environment, and notes. | Change algorithm behavior. | `tests/test_phase1_real_io_observability.py` |
| Experiment workflows | `scripts/workflow/` | Registers method lifecycles, expands ablation units, records manifest aliases, plans explicit low-level commands, and prunes completed command prefixes from live status evidence. | Put run-directory or resume state into runtime retriever classes. | `tests/test_experiment_runner.py`, `tests/test_workflow_orchestration.py` |

## File Map

| Area | Files | What to review |
|---|---|---|
| Experiment runner | `scripts/experiment.py`, `scripts/workflow/`, `graph_memory/experiment.py`, `configs/experiments/*.json`, `configs/search_spaces/*.json` | Manifest paths, config precedence, workflow registration, stage planning, method filtering, status/stale detection. |
| CLI adapters | `scripts/prepare_hotpotqa.py`, `scripts/build_graphs.py`, `scripts/run_retrieval.py`, `scripts/tune_graph_rerank.py`, `scripts/build_train_pairs.py`, `scripts/train_method.py`, `scripts/evaluate_retrieval.py`, `scripts/aggregate_tables.py` | Argument names, config visibility, validation calls, run summaries, output paths. |
| Contracts and validation | `graph_memory/contracts/`, `graph_memory/validation/` | Field names, forbidden fields, strict invariants, readable type annotations. |
| Data conversion | `graph_memory/datasets/`, `graph_memory/text/` | Stable task IDs, supporting-fact mapping, split determinism, label separation, text/entity behavior. |
| Graph construction | `graph_memory/graphs/` | Edge semantics, edge limits, deterministic sorting, graph views, no label access. |
| Retrieval | `graph_memory/stages/retrieve.py`, `graph_memory/registry/retrieval*.py`, `graph_memory/retrieval/catalog.py`, `graph_memory/retrieval/`, `graph_memory/retrieval_registry.py` | Stage orchestration, method capability projections, complete rankings, method construction, graph-method requirements. |
| Train pairs | `graph_memory/training_pairs/` | Sampling order, random seed behavior, positive/negative invariants. |
| Trainable model | `graph_memory/models/graph_retriever/` | Tensorization, feature order, neural model construction, checkpoint schema, inference boundary. |
| Evaluation | `graph_memory/evaluation/` | Metric definitions, exact joins, shared-graph connectivity, N/A path metrics. |
| Operations | `graph_memory/io.py`, `graph_memory/observability.py`, `graph_memory/training_config.py`, configs | Thin workflow-facing ports, deterministic writes, config defaults, run summary fields. |

## Review Checklist

- Input artifacts contain no label-only fields.
- Graph construction reads only `*_memory_tasks.input.json` fields.
- Retrieval methods return complete rankings over every memory node.
- Graph-rerank components consume explicit seed scores and graph structure.
- Graph-rerank tuning reuses seed-retriever scores across candidate configs without writing a persistent score-cache artifact.
- Graph-rerank configs use `neighbor_type_weights`; old `type_weights` artifacts must be converted before reuse.
- Evaluation reads labels from label artifacts only.
- Dev tuning and test evaluation are separate.
- Trainable retrieval inference reads task inputs, graphs, and checkpoint only.
- Every script writes a run summary when output paths are known.
- Experiment-runner paths stay under `runs/<experiment_name>/`.
- `scripts/experiment.py plan` renders explicit low-level commands instead of hiding artifact contracts; use `--no-cache` to render the full selected plan when the default cache-aware prefix pruning would hide completed commands.
- Root `graph_memory` integration ports remain thin and limited to workflow-facing APIs.
- `docs/40-operations/commands.md` matches actual script arguments.
- Tests pass or skip only for documented local-model reasons.

## Test And Verification

Use repository-local Python on this Windows host:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q --basetemp C:\tmp\graph-memory-core-refactor -p no:cacheprovider
uv run basedpyright --outputjson --level error
uv run ruff check
openspec validate --all --strict
```

If `uv run` cannot access the local uv cache, use the repository `.venv` for pytest and retry `uv` commands with the appropriate local permissions.

Architecture and import boundaries are covered by:

```text
tests/test_core_refactor_final_boundaries.py
tests/test_retrieval_domain_boundaries.py
tests/test_trainable_graph_domain_boundaries.py
```

## Extension Notes

- Add a new retriever runtime implementation under `graph_memory/retrieval/methods/`, register public metadata in `graph_memory/registry/retrieval.py`, and wire construction through `graph_memory/registry/retrieval_builders.py`.
- If it uses an existing experiment lifecycle, add one static registration in `scripts/workflow/registry.py`.
- If it has a genuinely new lifecycle, add one local adapter in `scripts/workflow/workflows.py` and register it. Do not add planner branches.
- Add a new flat score-based baseline by adding a seed `Retriever` and a `RetrievalMethod` wrapper under `graph_memory/retrieval/methods/flat/`.
- Add a new graph reranker by keeping the boundary `initial_scores + graph + config -> complete ranking` and adding graph-rerank pieces under `graph_memory/retrieval/methods/graph_rerank/`.
- Add GraphRAG, MemGPT-style, or trainable graph methods as separate `RetrievalMethod` implementations if their core behavior is traversal, hierarchy selection, or learned message passing rather than a weighted score sum.
- Add graph ablations by extending `GraphBuildConfig` or adding named graph-transform functions before retrieval.
- Add a new dataset converter by producing the same `MemoryTaskInput` and `MemoryTaskLabels` artifacts.
- Add new metrics by introducing pure metric primitives first, then adding aggregate columns in `evaluate_results` and table split helpers.
- Add new experiment defaults under `configs/experiments/`; keep tuning grids under `configs/search_spaces/` and curated result configs under `configs/published/`.

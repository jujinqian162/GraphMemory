## 1. Tests First: Preserve Existing Runner Behavior

- [x] 1.1 Run the current focused runner and R-GCN test set with an isolated pytest temp directory and record the pre-refactor baseline, without modifying unrelated dirty report or test files. Baseline: `53 passed`.
- [x] 1.2 Add regression tests proving that ablation-disabled BM25, graph-rerank, and R-GCN manifests retain their current deterministic paths and method-first stage sequences.
- [x] 1.3 Add regression tests proving that existing `scripts/experiment.py init`, `plan`, `run`, `status`, resource discovery, and `--from` / `--to` CLI behavior remain available when ablation is disabled.
- [x] 1.4 Add a compatibility test proving that documented low-level scripts remain directly callable without a manifest or workflow package.

## 2. Tests First: Typed Workflow Contracts

- [x] 2.1 Add failing tests for enum-backed closed values: stages, workflow IDs, artifact roles, change dimensions, artifact states, and R-GCN variant IDs.
- [x] 2.2 Add failing tests that invalid CLI, config, or manifest control values fail fast and report allowed enum values.
- [x] 2.3 Add failing workflow-registry validation tests proving every runtime retrieval method resolves exactly one workflow adapter and every suite references a registered method.
- [x] 2.4 Add a failing extension test with a test-only method registration proving that a same-lifecycle method can reuse an existing workflow without adding planner branches.

## 3. Tests First: Invalidation, Aliases, And Resume

- [x] 3.1 Add failing tests proving `MODEL_STRUCTURE` and `MODEL_GRAPH_VIEW` variants invalidate R-GCN from `train` while aliasing main graph and train-pair artifacts.
- [x] 3.2 Add failing tests proving `PAIR_SAMPLING` variants invalidate R-GCN from `pairs`, allocate variant-local pair artifacts, and allocate all downstream outputs locally.
- [x] 3.3 Add failing tests proving `full_rgcn` aliases ordinary main R-GCN checkpoint, prediction, and metrics artifacts without scheduling duplicate lifecycle commands.
- [x] 3.4 Add failing tests proving `--from retrieve` schedules only downstream commands when all selected variant checkpoints exist.
- [x] 3.5 Add failing tests proving `--from retrieve` reports missing variant checkpoint paths and never silently inserts `train`.

## 4. Add The Typed Workflow Package

- [x] 4.1 Create `scripts/workflow/__init__.py` and `scripts/workflow/types.py` with bilingual documented enums and typed records for stages, workflows, artifacts, dimensions, states, run units, variants, commands, and status rows.
- [x] 4.2 Create `scripts/workflow/registry.py` with static method-to-workflow registrations, suite registrations, validation helpers, and discovery helpers.
- [x] 4.3 Create `scripts/workflow/workflows.py` with small ordered adapters for stateless retrieval, graph rerank, and R-GCN trainable retrieval.
- [x] 4.4 Ensure workflows declare semantic input roles, output roles, invalidating dimensions, and low-level command adapters without inspecting filesystem status or outer planner state.

## 5. Move Manifest And Artifact Ownership

- [x] 5.1 Create `scripts/workflow/artifacts.py` with deterministic main and variant artifact namespaces plus explicit alias records.
- [x] 5.2 Create `scripts/workflow/manifest.py` by moving manifest initialization, loading, effective-config freezing, training-config writing, and path generation out of `graph_memory/experiment.py`.
- [x] 5.3 Revise new manifests to record run units, selected variants, suite metadata, artifact aliases, ablation metric-index path, and ablation table path.
- [x] 5.4 Keep schema-version-1 manifests readable as non-ablation runs and reject in-place ablation enablement without explicit reinitialization.

## 6. Implement Workflow-Driven Planning

- [x] 6.1 Create `scripts/workflow/planner.py` and move stage-range selection, dependency validation, concrete command planning, formatting, and execution ordering out of `graph_memory/experiment.py`.
- [x] 6.2 Implement run-unit expansion from selected methods and registered suites without adding an `ablate` stage.
- [x] 6.3 Implement earliest-invalidated-step calculation from `VariantSpec.changed_dimensions` and `WorkflowStepSpec.invalidated_by`.
- [x] 6.4 Implement upstream artifact aliasing and downstream variant-local allocation from the calculated invalidation boundary.
- [x] 6.5 Preserve fail-fast dependency checks for tuned graph configs, pairs, and checkpoints when required artifacts are neither scheduled nor valid.
- [x] 6.6 Deduplicate shared `prepare` and `graphs` commands across ordinary methods and ablation run units by semantic artifact reference.
- [x] 6.7 Render variant-qualified command blocks while preserving explicit low-level script paths and arguments.

## 7. Implement Status Inspection

- [x] 7.1 Create `scripts/workflow/status.py` and move artifact evidence inspection plus status formatting out of `graph_memory/experiment.py`.
- [x] 7.2 Report ordinary and variant-qualified stage status with `missing`, `complete`, `stale`, and `alias` states where evidence is available.
- [x] 7.3 Preserve current retrieval-summary stale detection and extend it to variant prediction artifacts.

## 8. Wire The CLI And Compatibility Facade

- [x] 8.1 Update `scripts/experiment.py` to import orchestration behavior from `scripts.workflow` while keeping the existing entry-point path and current subcommands.
- [x] 8.2 Add `ablations list` discovery with optional `--method`, including variant IDs and changed dimensions.
- [x] 8.3 Add repeated `--variant` filters and `--ablations-only` to `plan` and `run`, with fail-fast validation against registered suites.
- [x] 8.4 Reduce `graph_memory/experiment.py` to a thin transitional compatibility facade that re-exports supported runner helpers and contains no method-specific planner branches.

## 9. Add Config-Controlled R-GCN Ablations

- [x] 9.1 Extend experiment config resolution with typed `enable_ablation` and optional ordered `ablation_variants` selection while preserving ablation-disabled defaults.
- [x] 9.2 Register the R-GCN suite with `full_rgcn`, `wo_bridge`, `wo_entity_overlap`, `wo_sequential`, `wo_query_overlap`, `wo_graph`, `wo_edge_type`, `wo_edge_weight`, `wo_seed_score`, and `wo_hard_negatives`.
- [x] 9.3 Generate each non-baseline effective training config as resolved main training config plus its registered minimal override and persist it under the variant namespace.
- [x] 9.4 Retire or rewrite the duplicated `configs/training/dense_rgcn_graph_retriever/ablations.json` template so `base.json` remains the single hand-maintained full training config.
- [x] 9.5 Add tests proving optimization, encoder, and unaffected sections remain identical to the main resolved config for each override category.

## 10. Complete Low-Level R-GCN Variant Support

- [x] 10.1 Add failing model-config and tensorization tests for `wo_entity_overlap`, `wo_sequential`, and `wo_query_overlap`.
- [x] 10.2 Extend R-GCN model-config construction so each edge-view variant removes exactly its declared model-visible edge type while sharing the normal training path.
- [x] 10.3 Add failing pair-generation tests for `wo_hard_negatives` and apply its minimal pair-sampling override without changing model or optimization settings.
- [x] 10.4 Add a failing inference test proving an edge-view variant's returned `retrieved_subgraph.edges` excludes hidden edge types.
- [x] 10.5 Reuse the model-visible edge policy when constructing R-GCN prediction subgraphs so debug and efficiency artifacts reflect visible edges.

## 11. Add Ablation Aggregation

- [x] 11.1 Add failing aggregation tests for a run-local `config/ablation_metrics_index.json` containing explicit method, variant, and metrics paths, including the aliased `full_rgcn` row.
- [x] 11.2 Extend `scripts/aggregate_tables.py` with optional `--ablation_index` and `--output_ablation` arguments while preserving existing non-ablation invocation.
- [x] 11.3 Write `tables/ablation_results.csv` with stable `Method`, `Variant`, `Recall@5`, `Full Support@5`, `Connected Evidence Recall@10`, `Path Recall@10`, and `Retrieval Latency / Query` columns.
- [x] 11.4 Document and test that existing connectivity columns retain reference-graph semantics across variants.

## 12. Integration And Documentation

- [x] 12.1 Add a smoke integration test that initializes an ablation-enabled R-GCN run with a small variant subset and verifies the expanded plan, aliases, generated configs, status rows, and ablation table command.
- [x] 12.2 Update experiment configs with documented `enable_ablation` defaults and a server-ready example for running selected R-GCN variants.
- [x] 12.3 Update `docs/configs/`, `docs/40-operations/commands.md`, `docs/40-operations/reproducibility.md`, and `docs/40-operations/implementation-handoff.md` with config, CLI, artifact layout, resume, and extension guidance.
- [x] 12.4 Document the extension rule: same-lifecycle retrievers register against an existing workflow; genuinely new lifecycles add one local adapter under `scripts/workflow/`; planner branches are not added.

## 13. Verification

- [x] 13.1 Run focused workflow-planner, experiment-runner, R-GCN tensorization, R-GCN inference, pair-generation, and aggregation tests with an isolated pytest temp directory.
- [x] 13.2 Run the full repository test suite with an isolated pytest temp directory.
- [x] 13.3 Run `ruff check graph_memory scripts tests`.
- [x] 13.4 Run `git diff --check`.
- [x] 13.5 Run `openspec validate refactor-experiment-workflow-ablation --strict`.
- [x] 13.6 Review the final diff and confirm no unrelated dirty report assets, report prose, or user test edits were reverted.

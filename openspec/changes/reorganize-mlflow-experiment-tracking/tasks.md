## 1. Lock the Replacement Contracts with Failing Tests

- [x] 1.1 Add a real Hydra 1.3.4 multirun spike proving the configured formatter can produce `0_num_layers=2`, `1_num_layers=3`, and `2_num_layers=4` as both Hydra output directories and `RunLayout` directories.
- [x] 1.2 Add failing layout tests for deterministic multi-dimension leaves, excluded fixed identity overrides, complete persisted overrides, illegal path characters, and duplicate collapsed leaf-key rejection.
- [x] 1.3 Replace tests that require one child per stage attempt with failing assertions for one child per selected baseline and zero prepare/graph/aggregate children.
- [x] 1.4 Add failing parent projection tests proving concise parameters, no `method_configs.*` or search-space parameters, no parent metrics, no parent comparison images, and organized config/result/workflow artifacts.
- [x] 1.5 Add failing child projection tests for method-relative parameters, common `final.*` keys, trainable `train.*` epoch series, organized baseline artifacts, and prohibited large uploads.
- [x] 1.6 Add failing metric-hygiene tests proving counts, timings, artifact sizes, duplicate aggregate rows, unknown numeric final columns, and non-numeric/NaN/infinite values do not enter the model-metric namespace.
- [x] 1.7 Add failing cache, exact-resume, method-failure, shared-stage-failure, hidden-dependency, explicit-dependency, ablation-variant, and stage-bounded-run tracking tests.
- [x] 1.8 Add a locked MLflow 3.14 spike for `mlflow.note.content` result-table rendering expectations and document the accepted plain-text fallback while retaining CSV artifacts.

## 2. Implement Concise Multirun Identity

- [x] 2.1 Implement one pure concise-override formatter that excludes configured root identity keys, collapses nested paths to leaf keys, preserves deterministic assignment order, sanitizes path values, and rejects leaf collisions.
- [x] 2.2 Register the formatter for Hydra before composition and update `hydra.sweep.subdir` so Hydra and `RunLayout` resolve the same concise directory without a second output tree.
- [x] 2.3 Update `MultirunIdentity`, `RunLayout`, Hydra initialization, and existing-job selection to use the concise suffix directly with no old full-override fallback.
- [x] 2.4 Update multirun manifest, status, inspect, reset, and command tests to list and accept concise job selectors while retaining full resolved YAML and override files.

## 3. Curate Parent Runs

- [x] 3.1 Replace recursive parent config flattening with an explicit parameter projection for name, dataset, profile, seed, device, methods, top-k, stage bounds, cache, ablation, and multirun identity only.
- [x] 3.2 Move resolved config and overrides to parent artifact paths `config/resolved.yaml` and `config/overrides.yaml` and remove their repeated upload from stage tracking.
- [x] 3.3 Render successful aggregate main/path/efficiency/ablation rows into one readable all-baseline parent Overview description without publishing native parent metrics.
- [x] 3.4 Upload available aggregate tables under `results/` and shared prepare/graph/aggregate status and summaries under `workflow/`, including coherent partial/failure behavior.
- [x] 3.5 Prove parent tracking creates no repository-generated comparison image, chart configuration, training series, final metric, method-specific parameter tree, or duplicated baseline artifact.

## 4. Replace Stage Children with Baseline Children

- [x] 4.1 Define the user-visible baseline identity from selected method plus optional variant, independently of expanded hidden training dependencies.
- [x] 4.2 Add direct `TrackingAdapter` operations to create, find, update, and terminate a unique child by parent and baseline identity tags.
- [x] 4.3 Replace per-invocation child creation in execution with a baseline-child ID map while preserving the existing stage-major subprocess plan and first-failure stop.
- [x] 4.4 Route pair, tune, train, retrieve, and evaluate observations to the owning child; route prepare, graph, and aggregate observations to the parent artifact/summary projection.
- [x] 4.5 Terminate each baseline child after its final owned observation, fail the owning child on a method-stage error, terminate all open children on parent failure, and avoid synthetic children after an early shared-stage failure.
- [x] 4.6 Create and populate selected baseline children from authoritative local summaries and result artifacts when every owned stage is a cache hit.
- [x] 4.7 Reuse the unique existing baseline child on exact resume and fail strictly on duplicate matching children without making MLflow part of cache validity.

## 5. Publish Comparable Baseline Data

- [x] 5.1 Implement and test the explicit canonical-column to method-independent `final.*` metric mapping for all numeric main, path, connectivity, and efficiency results.
- [x] 5.2 Log each applicable final value exactly once on its baseline child, omit numeric logging for `N/A`, and delete dynamic method/stage/variant-prefixed aggregate metric generation.
- [x] 5.3 Preserve epoch-indexed `train.*` series from trainable methods, including common loss/development fields and method-specific numeric observations, while rejecting nested or non-finite metric values.
- [x] 5.4 Publish only the selected method's effective encoder, pair, tuning, model, trainer, and scoring configuration under concise method-relative child parameters.
- [x] 5.5 Move counts, timings, status, errors, artifact paths, kinds, and sizes to tags, parameters, Overview text, or summary artifacts rather than MLflow model metrics.
- [x] 5.6 Organize child artifacts under `config/`, `tuning/`, `training/`, `evaluation/`, `workflow/`, and `dependencies/` and allowlist small training/evaluation files under the existing size ceiling.
- [x] 5.7 Preserve deny-by-default metadata-only handling for datasets, graphs, train pairs, predictions, checkpoints, model directories, and other large scientific outputs.

## 6. Handle Dependencies, Variants, Failures, and Resume

- [x] 6.1 Attach a hidden Dense-FT prerequisite's summaries and artifact references to `dense_ft_rgcn_graph_retriever/dependencies/dense_ft/` without creating an extra child.
- [x] 6.2 When Dense-FT is explicitly selected, give it its own child and make the dependent baseline reference rather than duplicate its tracked observations.
- [x] 6.3 Treat each executable ablation variant as a distinct baseline identity with the same final metric keys and variant-specific parameters/artifacts.
- [x] 6.4 Verify interrupted resume appends newly available observations to existing baseline children without duplicate metric publication or duplicate children.
- [x] 6.5 Verify cache-disabled reruns and repeated stage attempts preserve one baseline child per selected identity and leave attempt-level detail in local stage summaries.
- [x] 6.6 Verify tracking write failures still fail the owning local attempt and cannot turn output existence or MLflow status into cache truth.

## 7. Remove the Former Projection and Compatibility Surfaces

- [x] 7.1 Delete per-stage MLflow child creation/termination paths, stage-prefixed final metric construction, recursive parent method-config flattening, and repeated resolved-config/override uploads.
- [x] 7.2 Remove or rewrite tests and documentation that promise stage-attempt children, cache-hit child suppression, parent result metrics, exhaustive numeric CSV mirroring, or full override directory leaves.
- [x] 7.3 Run repository-wide scans proving there is no tracking schema/version tag, compatibility reader, legacy tracking branch, dual-write path, old-run migration, database rewrite, or generated parent comparison plot.
- [x] 7.4 Document the direct cutover rule: existing SQLite rows remain historical, and former local run names require a new name or explicit normal reset before execution.

## 8. Documentation and Acceptance Workflows

- [x] 8.1 Update README and operations commands with concise multirun directories, parent/baseline responsibilities, baseline parameter search, artifact paths, and child-based MLflow Compare Runs steps.
- [x] 8.2 Update reproducibility, logging, implementation handoff, and active design documentation so local summaries remain authoritative and MLflow is the parent/baseline presentation mirror.
- [x] 8.3 Run a fresh-name seven-baseline workflow and verify one concise parent, exactly seven children, a readable parent result table, total artifacts, common final keys, and trainable curves.
- [x] 8.4 Run a fresh three-job `num_layers=2,3,4` multirun and verify concise directories/names plus direct Compare Runs compatibility across the three R-GCN baseline children.
- [x] 8.5 Run fresh-name non-trainable-only, trainable-only, hidden-dependency, ablation, fully cached, interrupted-resume, cache-disabled, partial-range, and expected-failure workflow acceptance cases.
- [x] 8.6 Verify the shared SQLite backend and artifact root remain fixed, existing rows are untouched, parent metrics remain empty, and no repository-generated comparison plot exists.
- [x] 8.7 Run the focused and full pytest suites, locked Python 3.10 Hydra/MLflow runtime smoke, Ruff, basedpyright, compileall, `git diff --check`, residual scans, and strict OpenSpec validation.

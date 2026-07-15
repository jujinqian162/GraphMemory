## 1. Raw Source Artifact Contract

- [x] 1.1 Add failing config/planner tests for explicit file and directory dataset source bindings
- [x] 1.2 Add required `source_kind` fields to dataset config models, resolved config, planner bindings, and all existing dataset YAML files
- [x] 1.3 Verify existing file-backed dataset plans remain behaviorally unchanged apart from explicit artifact identity

## 2. TRAJECT-Bench Domain Model

- [x] 2.1 Add typed raw query, tool catalog, ranking, label, and conversion records under `graph_memory/datasets/traject_bench`
- [x] 2.2 Implement deterministic official partition discovery for parallel/simple, parallel/hard, and sequential files
- [x] 2.3 Implement strict query/tool-call parsing for official parallel and sequential schema variants
- [x] 2.4 Implement tool-catalog parsing and deterministic duplicate-name canonicalization
- [x] 2.5 Implement stable tool IDs, catalog-only candidate text, distinct gold sets, repeated gold sequences, and self-loop-free dependency conversion
- [x] 2.6 Add parser/converter tests for duplicates, repeated calls, deterministic IDs, and missing-gold rejection

## 3. Retrieval and Validation Boundaries

- [x] 3.1 Implement TRAJECT-Bench text ranking, EvidenceGraph build/ranking, and evidence evaluation projectors
- [x] 3.2 Project only catalog-visible connections into graph inputs and test that gold trajectory order cannot enter ranking/graph requests
- [x] 3.3 Add TRAJECT-Bench ranking/label validators including candidate identity, metadata, sequence, edge, and cross-artifact checks
- [x] 3.4 Register the dataset in selection, validation, stage, and type dispatch without loose string fallbacks
- [x] 3.5 Add end-to-end request projection and evaluation tests for BM25/Dense-compatible records

## 4. Prepare and Experiment Workflow

- [x] 4.1 Add `scripts/prepare_traject_bench.py` with invalid-before-sampling filtering and detailed catalog/query counts
- [x] 4.2 Add `configs/dataset/traject_bench.yaml` with pinned-revision capacities and directory source paths
- [x] 4.3 Add prepare-stage tests for split discovery, deterministic sampling, invalid reason reporting, and output separation
- [x] 4.4 Add planner tests showing frozen TRAJECT-Bench baselines avoid train/pair/evidence-graph stages unless method metadata requires them

## 5. Operations and Scope Documentation

- [x] 5.1 Document the dataset contract, operational split mapping, invalid-upstream policy, and retrieval-only metric interpretation
- [x] 5.2 Document pinned `hf download`, official GitHub alternative, smoke, quick BM25, and quick Dense server commands
- [x] 5.3 Update dataset/config command indexes so TRAJECT-Bench is discoverable through the normal experiment CLI

## 6. Official-Data and Quality Verification

- [x] 6.1 Validate OpenSpec artifacts strictly and run focused adapter/config/planner tests
- [x] 6.2 Prepare smoke records from the pinned official TRAJECT-Bench data and inspect counts plus label-leakage invariants
- [x] 6.3 Run a real BM25 workflow smoke through prepare, retrieve, evaluate, and aggregate
- [x] 6.4 Run a real Dense workflow smoke through prepare, retrieve, evaluate, and aggregate
- [x] 6.5 Run the full test suite, Ruff, basedpyright error gate, compileall, boundary scans, and `git diff --check`

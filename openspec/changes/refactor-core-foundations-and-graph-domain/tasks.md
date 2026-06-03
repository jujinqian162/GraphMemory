## 1. Batch 0 - OpenSpec and Baseline Freeze

- [x] 1.1 Record the current pytest, type-check, and OpenSpec strict-validation baseline commands before moving production code
- [x] 1.2 Add parser contract tests for affected public scripts without comparing formatted help text
- [x] 1.3 Add workflow planning contract tests for manifest fields, stage order, command arguments, method/profile narrowing, ablation selection, and `--ablations-only` fail-fast behavior
- [x] 1.4 Add tiny deterministic fixtures for foundation/domain behavior that Change A will later move
- [x] 1.5 Run the Batch 0 focused tests and validation commands and keep production code unmoved

## 2. Batch 1 - Contracts, Validation, and Infrastructure

- [x] 2.1 Create `graph_memory/contracts/` modules for artifact-shaped task, graph, ranking, training-pair, metric, and observability records
- [x] 2.2 Split artifact and config validators into `graph_memory/validation/` modules without changing fail-fast behavior
- [x] 2.3 Move IO and run-summary implementation into `graph_memory/infrastructure/`
- [x] 2.4 Keep root `graph_memory/io.py` and `graph_memory/observability.py` as narrow workflow integration ports only
- [x] 2.5 Update imports for migrated foundation modules and verify no new migrated-domain imports are added from `graph_memory.types`

## 3. Batch 2 - Dataset and Text Domains

- [x] 3.1 Move HotpotQA parsing, compatibility conversion, and split helpers into `graph_memory/datasets/`
- [x] 3.2 Move tokenization, lexical scoring, and entity helpers into `graph_memory/text/`
- [x] 3.3 Update scripts and tests to import the new dataset/text modules
- [x] 3.4 Verify parsing errors, conversion order, tokenization, IDF, lexical score, and entity outputs remain stable

## 4. Batch 3 - Graph Domain

- [x] 4.1 Move graph config, graph index, statistics, and graph views into `graph_memory/graphs/`
- [x] 4.2 Extract graph construction into explicit builder, prepared-input, edge-accumulator, and edge-rule modules
- [x] 4.3 Preserve edge rule order, edge weights, deduplication, bridge behavior, and graph statistics
- [x] 4.4 Update scripts and tests to import the graph domain modules and run graph-focused regression tests

## 5. Batch 4 - Evaluation Domain

- [x] 5.1 Move metric primitives, connectivity derivation, evaluation service, table rows, and failure-case generation into `graph_memory/evaluation/`
- [x] 5.2 Preserve metric definitions, task joins, fail-fast behavior, CSV columns, and failure-case output
- [x] 5.3 Update scripts and tests to import the evaluation domain modules
- [x] 5.4 Run focused evaluation tests plus full Change A validation gates

## 1. Contracts and configuration

- [x] 1.1 Add the `cross_encoder` method ID and flat/provenance-unit closed config.
- [x] 1.2 Add matched profile settings and an ISETrace-only compatibility check.
- [x] 1.3 Extend pair/cache, inspection, normalized config, and output identities.

## 2. Training and checkpoint lifecycle

- [x] 2.1 Add Cross-Encoder metadata with strict method/variant validation.
- [x] 2.2 Build labelled query-candidate examples from existing pair artifacts.
- [x] 2.3 Train one-logit BCE models with deterministic seeds and dev Recall@5 selection.
- [x] 2.4 Materialize model and training-metric artifacts with scientific source digests.

## 3. Retrieval and workflow

- [x] 3.1 Implement complete task-local Cross-Encoder scoring and deterministic ties.
- [x] 3.2 Wire train/dev/test preparation, graph-free pair building, training, ranking, evaluation, and delivery.
- [x] 3.3 Enforce flat/provenance-unit checkpoint matching and graph isolation.
- [x] 3.4 Provision the frozen registered backbone without adding a test-set model search.

## 4. Tests and documentation

- [x] 4.1 Add config, candidate-view, cache, metadata, scorer, and workflow tests.
- [x] 4.2 Add smoke commands and document full-candidate ranking, seeds, outputs, and interpretation.
- [x] 4.3 Run focused/full pytest, Ruff, BasedPyright, compileall, diff check, and strict OpenSpec validation.

## 1. Restore Method-Only Scope

- [x] 1.1 Restore the existing twowiki_provenance schema, converter, parser, scoring, projectors, dataset config, fixture, prepare/cache behavior, and dataset tests.
- [x] 1.2 Restore shared evaluation metrics, evaluation artifacts, aggregate outputs, and remove paired/counterfactual/audit tooling introduced by the broader implementation.
- [x] 1.3 Restore trainable provenance R-GCN inference/tests and ensure no training method, pair, checkpoint, or model-cache contract is changed.
- [x] 1.4 Rewrite the source implementation plan and operations documentation so they state the strict method-only boundary.

## 2. Improve Execution-Provenance Retrieval

- [x] 2.1 Keep strict bounded-search, confidence-gate, protected-prefix, and hop-penalty configuration under the existing public method ID.
- [x] 2.2 Use existing request-graph edge weights exactly once while treating binding, completeness, and lifecycle as boolean validity gates.
- [x] 2.3 Produce at most one deterministic partner per original Dense seed and enforce beam/hop/expansion limits.
- [x] 2.4 Resolve partner conflicts deterministically and perform protected-prefix stable insertion while preserving non-promoted Dense order.
- [x] 2.5 Preserve original Dense objects/scores on exact fallback and emit only candidate edges for real promotions.

## 3. Improve GraphRAG

- [x] 3.1 Keep typed title/body entity evidence, deterministic normalization, alias ambiguity rejection, and hub suppression local to GraphRAG.
- [x] 3.2 Keep the private batched frozen-Dense title-group resolver with singleton, tie, and margin-abstention behavior.
- [x] 3.3 Reuse one encoder instance for Dense ranking and sentence resolution without adding a shared orchestration layer.
- [x] 3.4 Replace global PPR/fusion with at most one pair-local partner per anchor, deterministic conflicts, protected-prefix insertion, and exact Dense fallback.
- [x] 3.5 Emit only directed candidate bridges that caused real promotions.

## 4. Keep Contracts and Verification Focused

- [x] 4.1 Keep only method-native trace fields and validation needed to audit proposal gates, conflicts, displacement, emitted edges, and exact fallback.
- [x] 4.2 Migrate only the two method YAML/config/builder paths; preserve dataset and workflow contracts.
- [x] 4.3 Retain focused behavior tests for both methods and remove clean-dataset/general-evaluation tests from this change.
- [x] 4.4 Run focused tests, full pytest, Ruff, basedpyright, compileall, git diff --check, and strict OpenSpec validation.

## 1. Freeze Failing Contracts

- [x] 1.1 Extend `tests/test_twowiki_provenance_dataset.py` with failing schema-v3 cases for gold-spine inclusion, fixed head/branch out-degree, non-gold rank-bucket matching, deterministic conversion, and typed rejection when a match cannot be formed.
- [x] 1.2 Add failing converter/tensor tests proving feed weights have the configured floor and source-mass sum and proving `wo_edge_weight` preserves endpoints, relation IDs, directions, and total source feed mass.
- [x] 1.3 Add failing provenance pair tests for native successor/predecessor negatives, cross-sampler deduplication, hardness precedence, overlap summaries, and easy-only `wo_hard_negatives` output.
- [x] 1.4 Add failing model/inference tests for task-balanced pairwise candidate loss, bounded stable promotion, conflict/no-op behavior, `wo_edge_rerank`, and threshold-based edge abstention.
- [x] 1.5 Add failing evaluation/checkpoint tests for Edge Precision/F1 arithmetic, zero-edge behavior, joint dev selection/tie-breaking, v2/v3 schema rejection, and variant lifecycle provenance.

## 2. Implement Schema-V3 Graph Construction

- [x] 2.1 Bump `TWOWIKI_PROVENANCE_SCHEMA_VERSION` and update dataset record types/parser errors so v2 raw records fail explicitly and v3 records carry versioned branch/confidence metadata.
- [x] 2.2 Extend `ProvenanceGraphConstructionConfig` and converter CLI with scorer identity, query-template version, hybrid weight, semantic temperature, weight floor, and explicit near/mid/tail rank buckets; include every field in construction identity.
- [x] 2.3 Implement complete source-aware hybrid ranking with batched frozen-dense reuse and deterministic BM25/dense/hybrid tie-breaking while retaining BM25-only and dense-only interventions.
- [x] 2.4 Replace fallback top-k repair with gold-spine-first construction plus one semantic head and one deterministic rank-banded branch per source.
- [x] 2.5 Implement per-record non-gold rank-bucket matching for the gold edge and the typed `unmatched_gold_branch_bucket` rejection path without adding ranking-visible gold/fallback fields.
- [x] 2.6 Compute source-local confidence probabilities and bounded feed weights with floor `0.5`; keep non-feed edges at `1.0` and validate exact source feed-mass sums.
- [x] 2.7 Expand converter statistics and manifest output with branch roles/buckets, gold/non-gold rank and weight distributions, gold-head rate, source-mass checks, scorer/cache identity, and typed rejection counts.
- [x] 2.8 Tighten `graph_memory/validation/twowiki_provenance.py` so ranking records reject gold-only metadata, unmatched rank buckets, non-fixed out-degree, invalid weight ranges/mass, incomplete confidence metadata, and schema/config mismatches.
- [x] 2.9 Update the committed smoke fixture/generator and converter-focused tests to schema v3, including deterministic shuffled candidate/edge order.

## 3. Add Provenance-Native Pair Supervision

- [x] 3.1 Extend train-pair sample contracts and validation with `hard_provenance_successor` and `hard_provenance_predecessor` without changing flat/EvidenceGraph method behavior.
- [x] 3.2 Introduce a typed provenance pair-build task and dataset projector carrying `TextRankingRequest`, `ExecutionProvenanceGraph`, and `EvidenceLabel`; route only the provenance R-GCN family through it in `graph_memory/stages/pairs.py`.
- [x] 3.3 Implement logical successor/predecessor samplers over contracted native transitions and expose explicit per-positive counts in the provenance method config.
- [x] 3.4 Deduplicate negatives by `(task_id, node_id)` across provenance/dense/BM25/easy sources using the declared precedence and record overlap, shortfall, and post-dedup counts in `summary.json`.
- [x] 3.5 Change the provenance full-profile pair defaults to successor/predecessor/dense/BM25/easy counts `2/1/1/1/2` per positive and make `wo_hard_negatives` zero every hard category while retaining easy samples.
- [x] 3.6 Verify planner/cache invalidation: model-only variants alias full pairs, `wo_hard_negatives` owns pairs and every downstream asset, and non-provenance methods retain their existing pair identity.

## 4. Align R-GCN Training and Weight Consumption

- [x] 4.1 Implement source-group mean resolution for `wo_edge_weight` in provenance tensorization and verify forward/reverse messages share the resolved mass-matched value.
- [x] 4.2 Replace provenance candidate BCE with task-balanced pairwise logistic ranking loss over every positive/unique-negative comparison while retaining class-balanced logical-edge BCE.
- [x] 4.3 Log candidate loss, edge loss, total loss, negative-category counts, and task/comparison counts per epoch so pair-source effects are inspectable.
- [x] 4.4 Preserve typed relation transforms and existing `wo_graph`/`wo_edge_type` computation, and add regression tests that each public model ablation changes exactly its declared signal.

## 5. Add Structured Candidate and Edge Inference

- [x] 5.1 Add strict checkpointed inference config for `structured_pool_size`, `structured_seed_top_s`, `preserve_node_top_n`, and `edge_accept_threshold` with initial values `16/5/2/0.5`.
- [x] 5.2 Implement deterministic per-seed best-successor selection, target conflict resolution, protected-prefix stable insertion, original-score-multiset reassignment, and movement/no-op traces.
- [x] 5.3 Replace forced successor emission with thresholded abstention and emit only retained logical/native edges whose endpoints survive final top-k.
- [x] 5.4 Register `wo_edge_rerank` as a ranking-stage ablation that aliases full pair/model artifacts, skips promotion, and retains raw edge diagnostics.
- [x] 5.5 Implement the explicit shuffled-feed-message diagnostic with deterministic endpoint permutation, unchanged non-feed messages and relation/weight histograms, and a distinct non-default cache identity.
- [x] 5.6 Bump the provenance checkpoint family to schema v3, serialize construction/pair/loss/weight/inference/selection identities and best component metrics, and reject v2/evidence checkpoints without compatibility fallback.

## 6. Complete Evaluation and Checkpoint Selection

- [x] 6.1 Add micro Edge Precision@10 and Edge F1@10 plus abstention rate to evaluation contracts, suites, metric validation, tables, MLflow mappings, and collected result deliveries.
- [x] 6.2 Replace `_dev_full_support` with full dev inference that returns Full Support@5, MRR, Edge Precision/Recall/F1@10, average emitted edges, and abstention rate under the effective variant.
- [x] 6.3 Implement `dev_joint = 0.50 * Full Support@5 + 0.25 * MRR + 0.25 * Edge F1@10` and deterministic component/earlier-epoch tie-breaking; store all values in training metrics and checkpoint metadata.
- [x] 6.4 Version v3 evaluation rows and make aggregation reject mixed v2/v3 schemas instead of filling or comparing missing edge columns.
- [x] 6.5 Add paired-analysis tooling or an existing-tool extension that outputs per-seed rows, mean/std, query-paired confidence intervals, and discordant Full Support counts without encoding an expected sign.

## 7. Wire Configuration, Migration, and Documentation

- [x] 7.1 Update the singular `method.variant` contract, method config, experiment config validation, Registry ablation metadata, planner invalidation boundaries, and inspection output for the five public provenance variants.
- [ ] 7.2 Generate pilot raw data under `data/twowiki_provenance/v3/raw`, preflight actual split counts, and switch `configs/dataset/twowiki_provenance.yaml` only after deterministic/leakage/weight audits pass.
- [x] 7.3 Ensure raw/prepared/pair/model/prediction/evaluation implementation identities prevent every v2 artifact from aliasing a v3 job; add no legacy translation or fallback branch.
- [x] 7.4 Update `docs/40-operations/twowiki-provenance.md` with v3 conversion/audit commands, full-dev requirement, current variant values, three-seed commands, edge-metric meanings, migration/rollback, and the synthetic-benchmark claim boundary.
- [ ] 7.5 Update paper-facing experiment notes only after formal results exist, and state that correctness gates do not require every ablation metric to be lower.

## 8. Verify and Run the Evidence Matrix

- [x] 8.1 Run focused dataset, provenance R-GCN, pair, evaluation, Registry/planner, artifact, and workflow tests outside the Windows sandbox and fix all failures.
- [ ] 8.2 Run `uv run ruff check`, `uv run basedpyright`, and the repository's broader required pytest gate outside the Windows sandbox.
- [x] 8.3 Convert the same v3 pilot twice and verify identical raw/manifest/statistics digests plus passing gold/non-gold bucket, metadata, and source-mass audits.
- [ ] 8.4 Run one quick seed-13 full/variant matrix, verify exact scientific-input digests and declared invalidation differences, and inspect promotion/abstention/pair-overlap traces before full training.
- [ ] 8.5 Freeze construction/model/threshold/selection settings from train/dev evidence, then run full-profile `full_rgcn`, `wo_graph`, `wo_edge_type`, `wo_edge_weight`, `wo_hard_negatives`, and `wo_edge_rerank` for seeds `13`, `17`, and `29` on the untouched test split.
- [ ] 8.6 Produce the v3 ablation delivery with every seed row, mean/std, paired intervals, discordant counts, Edge Precision/Recall/F1, average edges, abstention, and explicit artifact identities; report observed ordering without treating it as an implementation acceptance gate.

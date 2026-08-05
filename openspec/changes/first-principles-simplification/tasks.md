## 1. Freeze the Behavior and Deletion Boundary

- [x] 1.1 Record the baseline full-test result, production/test line counts, and module counts for experiment, stages, registry, retrieval, models, training pairs, and evaluation.
- [x] 1.2 Inventory every production and test caller of result envelopes/batches, execution tasks, retrieval registries/builders, training payloads/trainers, model factories, sampler protocols, and config projection helpers.
- [ ] 1.3 Record representative prepared-data, model metadata/checkpoint, ranking, metric, per-task, manifest, cache, and tracking shapes currently protected by tests.
- [x] 1.4 Add or identify behavior tests for all eight methods and both graph domains before deleting implementation-shape tests.
- [x] 1.5 Establish a per-tranche net-line-count check and require every completed tranche to reduce production plus test code after replacements.

## 2. Tranche 1 - Remove Pure Wrappers and Duplicate Outputs

- [x] 2.1 Replace `RankedResultEnvelope` and `RankedResultBatch` production uses with direct `RankedResult` assembly and one boundary consistency check.
- [x] 2.2 Delete tests that instantiate result wrappers solely to test repeated request/result context, replacing them with ranking behavior and persisted-result tests.
- [x] 2.3 Inline and delete single-use no-behavior runtime rows and graph scoring factory helpers proven unused by production callers; defer metric suite/service cleanup to the dedicated evaluation pass.
- [ ] 2.4 Remove combined dataset DTOs and sidecar outputs that have no workflow, script, or analysis consumer while retaining authoritative tasks, labels, graph, origin, and source-span data.
- [ ] 2.5 Collapse redundant result CSV projections to the retained authoritative outputs only after cache, delivery, tracking, docs, and analysis consumers are migrated.
- [ ] 2.6 Run focused preparation/ranking/evaluation/artifact tests and record Tranche 1 net deletion.

## 3. Tranche 2 - Replace Retrieval Framework with Direct Dispatch

- [ ] 3.1 Move concrete construction logic for BM25, Dense, Dense-FT, GraphRAG, provenance path, provenance R-GCN, and both evidence R-GCN methods behind one explicit retrieval-stage dispatch.
- [x] 3.2 Delete `RetrievalRegistry`, `RetrievalBuilderSpec`, settings-to-payload maps, and builder registration tables without introducing a replacement registry or plan.
- [x] 3.3 Delete `RetrievalExecutionTask`; pass each concrete request directly to the retrieval loop and keep task/candidate checks at assembly/persistence boundaries.
- [x] 3.4 Delete `ScorePipelineMethod`; make BM25 and Dense concrete rank functions/methods return the existing retrieval result directly while preserving Dense batch encoding.
- [ ] 3.5 Remove thin evidence trainable-graph adapters where direct `GraphRetrieverInference` calls preserve behavior; retain the provenance-specific tensorization adapter.
- [ ] 3.6 Remove registry/settings/payload exports and rewrite implementation-shape tests around all eight methods' ranking behavior and provenance metadata.
- [ ] 3.7 Run retrieval, graph inference, evaluation, config-composition, and full tests; record Tranche 2 net deletion.

## 4. Tranche 3 - Flatten Training and Configuration Projections

- [ ] 4.1 Make model materializers call concrete Dense-FT, evidence R-GCN, and provenance R-GCN training functions directly.
- [ ] 4.2 Delete `stages/train_payloads.py`, `stages/trainers.py`, trainer protocols, train dependency carriers, and single-implementation Dense-FT factories/runners that only forward arguments.
- [ ] 4.3 Inline graph scoring model construction and delete `GraphScoringModelFactory` while preserving checkpoint model configuration and inference loading.
- [ ] 4.4 Replace negative-sampler protocol/context/factory dispatch with direct calls to the existing four sampling algorithms inside pair construction.
- [ ] 4.5 Remove `effective`, train-stage, R-GCN train-stage, and ranking config projections that only copy method fields; make each concrete algorithm consume the single parsed owner.
- [ ] 4.6 Fix one-valued protocol choices at their concrete boundary, including the current optimizer/scheduler and selection-metric behavior, without changing numerical results.
- [ ] 4.7 Reduce checkpoints to inference-required state only where no resume or analysis consumer exists; document intentional invalidation of historical internal checkpoint schema.
- [ ] 4.8 Rewrite payload/factory/config-shape tests as Dense-FT and both graph-family training, selection, checkpoint-load, and seeded-R-GCN behavior tests.
- [ ] 4.9 Run training-pair, Dense-FT, R-GCN, config-composition, checkpoint, and full tests; record Tranche 3 net deletion.

## 5. Tranche 4 - Express Experiment Orchestration Once

- [ ] 5.1 Collapse repeated method branches in `experiment/workflow.py` into the direct prepare, optional graph/pairs/encoding, optional train, rank, evaluate, and output lifecycle.
- [ ] 5.2 Delete Prefect task transport wrappers that add no cache/resume boundary; retain only demonstrably required task decorators as thin direct calls.
- [ ] 5.3 Remove repeated method/stage/artifact binding authorities so workflow and stage materializers consume the same parsed config and filesystem paths.
- [ ] 5.4 Move input provisioning, MLflow recording, benchmark collection, inspection, and report publication outside method-specific scientific branches while preserving their current behavior.
- [ ] 5.5 Reduce prepared/model/result artifact type hierarchies without deleting digest, origin, revision, manifest, ranking, metric, per-task, failure-case, or model provenance required by current workflows.
- [ ] 5.6 Delete profile and stage fields that have one effective production value or no consumer; retain one formal protocol and test-only small execution controls.
- [ ] 5.7 Rewrite workflow/task/config/artifact filename tests around plan execution, cache/resume, tracking, final outputs, and all eight retained methods.
- [ ] 5.8 Run cache/resume, Prefect, MLflow, artifact, workflow, config-composition, and full tests; record Tranche 4 net deletion.

## 6. Data and Graph Domain Cleanup

- [ ] 6.1 Convert each evidence record once during preparation and reuse the successful conversion for validity filtering and sampling.
- [ ] 6.2 Inline dataset projector classes into dataset selection/preparation while preserving dataset-specific parsing and label derivation.
- [ ] 6.3 Remove `Converted*`, `ConversionResult`, and `Combined*` field-transport DTOs when the direct prepared task/label outputs cover their consumers.
- [ ] 6.4 Split ISETrace flat and provenance preparation into direct branches that reuse trajectory parsing but do not introduce a representation enum or universal prepared object.
- [ ] 6.5 Inline single-implementation evidence graph rule/builder/context scaffolding while preserving all four edge algorithms, deduplication, and the conversion script caller.
- [ ] 6.6 Inline single-implementation provenance extractors and remove unused query-synthesis catalog/selector exports while preserving source-span and motif semantics.
- [ ] 6.7 Run dataset label, ISETrace trajectory/span, graph construction, provenance tensorization, and full tests; record domain-cleanup net deletion.

## 7. Documentation and OpenSpec Consolidation

- [ ] 7.1 Update active architecture and operation docs to show the direct experiment, retrieval, training, and two-domain graph flows.
- [ ] 7.2 Remove active documentation for deleted wrappers, registries, factories, payloads, duplicate outputs, and projected config authorities.
- [ ] 7.3 Review existing active OpenSpec changes that require deleted internals; archive completed history or mark it superseded without copying those contracts into this change.
- [ ] 7.4 Verify docs and configs describe all eight retained methods but do not present optional methods as reasons for plugin architecture.

## 8. Final Verification

- [ ] 8.1 Run full pytest, Ruff, basedpyright at error level, compileall, and `git diff --check`.
- [ ] 8.2 Run strict OpenSpec validation and confirm every checked task has corresponding implementation or validation evidence.
- [ ] 8.3 Run retained dataset/method config composition and representative local smoke workflows for flat, Dense-FT, evidence graph, and provenance graph families.
- [ ] 8.4 Compare deterministic preparation, rankings, metrics, manifests, cache/resume outcomes, and tracking behavior against the baseline within existing tolerances.
- [ ] 8.5 Perform final call-site and export scans proving deleted wrappers, registries, payloads, factories, compatibility aliases, and duplicate authorities are absent.
- [ ] 8.6 Report production, test, docs/config, class, and module net reductions plus residual scientific and operational risks for review.

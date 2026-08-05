## 1. Freeze the Behavior and Deletion Boundary

- [x] 1.1 Record the baseline full-test result, production/test line counts, and module counts for experiment, stages, registry, retrieval, models, training pairs, and evaluation.
- [x] 1.2 Inventory every production and test caller of result envelopes/batches, execution tasks, retrieval registries/builders, training payloads/trainers, model factories, sampler protocols, and config projection helpers.
- [ ] 1.3 Record representative prepared-data, model metadata/checkpoint, ranking, metric, per-task, manifest, cache, and tracking shapes currently protected by tests.
- [x] 1.4 Add or identify behavior tests for all eight methods and both graph domains before deleting implementation-shape tests.
- [x] 1.5 Establish a per-tranche net-line-count check and require every completed tranche to reduce production plus test code after replacements.

## 2. Tranche 1 - Remove Pure Wrappers and Duplicate Outputs

- [x] 2.1 Replace `RankedResultEnvelope` and `RankedResultBatch` production uses with direct `RankedResult` assembly and one boundary consistency check.
- [x] 2.2 Delete tests that instantiate result wrappers solely to test repeated request/result context, replacing them with ranking behavior and persisted-result tests.
- [x] 2.3 Inline and delete single-use no-behavior runtime rows, graph scoring factory helpers, and the single-implementation node metric suite/factory; retain direct evidence/span evaluation functions.
- [x] 2.4 Remove combined dataset DTOs and `combined.json`; authoritative tasks, labels, graph, origin, and source-span outputs remain.
- [x] 2.5 Collapse redundant three-table result projections to the authoritative final metrics and per-task outputs; raw historical planning documents remain untouched.
- [x] 2.6 Run focused preparation/ranking/evaluation/artifact tests and record Tranche 1 net deletion.

## 3. Tranche 2 - Replace Retrieval Framework with Direct Dispatch

- [x] 3.1 Move concrete construction logic for BM25, Dense, Dense-FT, GraphRAG, provenance path, provenance R-GCN, and both evidence R-GCN methods behind one explicit static dispatch in `build_retrieval`.
- [x] 3.2 Delete `RetrievalRegistry`, `RetrievalBuilderSpec`, settings-to-payload maps, and builder registration tables without introducing a replacement registry or plan.
- [x] 3.3 Delete `RetrievalExecutionTask`; pass each concrete request directly to the retrieval loop and keep task/candidate checks at assembly/persistence boundaries.
- [x] 3.4 Delete `ScorePipelineMethod`; make BM25 and Dense concrete rank functions/methods return the existing retrieval result directly while preserving Dense batch encoding.
- [x] 3.5 Remove the thin evidence `TrainableGraphRetrievalMethod` adapter by making `GraphRetrieverInference` implement the retrieval protocol directly; retain the provenance-specific tensorization adapter.
- [x] 3.6 Remove retrieval registry/spec exports and rewrite implementation-shape tests around all eight methods' ranking behavior and provenance metadata; retain temporary settings/payload inputs until stage dispatch is inlined.
- [x] 3.7 Run retrieval, graph inference, evaluation, config-composition, and full tests; record Tranche 2 net deletion.

## 4. Tranche 3 - Flatten Training and Configuration Projections

- [x] 4.1 Make model materializers call concrete Dense-FT, evidence R-GCN, and provenance R-GCN training functions directly.
- [x] 4.2 Delete `stages/train_payloads.py`, `stages/trainers.py`, trainer protocols, train dependency carriers, single-implementation stage trainer wrappers, and unused public training injection hooks that only forward arguments.
- [x] 4.3 Inline graph scoring factory construction and delete `GraphScoringModelFactory` while preserving `build_model_from_config`, checkpoint model configuration, and inference loading.
- [x] 4.4 Replace negative-sampler protocol/context/factory dispatch with direct calls to the existing four sampling algorithms inside pair construction.
- [x] 4.5 Remove train-stage and ranking config projections that only copy method fields; concrete stage algorithms consume the parsed method configuration or its existing effective R-GCN subsection directly.
- [x] 4.6 Fix one-valued protocol choices at their concrete boundary, including the current AdamW/no-scheduler behavior, without changing numerical results.
- [x] 4.7 Reduce checkpoints to inference-required model state plus model/training provenance; invalidate the historical internal checkpoint schema as version 4.
- [x] 4.8 Rewrite payload/factory/config-shape tests as Dense-FT and both graph-family training, selection, checkpoint-load, and seeded-R-GCN behavior tests.
- [x] 4.9 Run training-pair, Dense-FT, R-GCN, config-composition, checkpoint, and full tests; record Tranche 3 net deletion.

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

- [x] 6.1 Convert each evidence record once during preparation and reuse the successful conversion for validity filtering and sampling.
- [x] 6.2 Inline all evidence and ISETrace projector classes into dataset selection while preserving dataset-specific parsing, request fields, graph-node metadata, and label derivation.
- [x] 6.3 Remove `Converted*`, `ConversionResult`, and evidence-dataset `*PreparedSplit` field-transport DTOs; direct ranking/label pairs cover all consumers.
- [ ] 6.4 Split ISETrace flat and provenance preparation into direct branches that reuse trajectory parsing but do not introduce a representation enum or universal prepared object.
- [x] 6.5 Inline the single-implementation evidence graph builder, context, accumulator, and rule classes into direct construction functions while preserving all four edge algorithms, deduplication, and the conversion script caller.
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

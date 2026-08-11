## 1. Freeze the variant contracts with failing tests

- [x] 1.1 Add config/inspection tests for `dense` and `dense_ft` flat defaults, provenance-unit overrides, matched scientific settings, and evidence-dataset rejection.
- [x] 1.2 Add ISETrace adapter tests proving provenance-unit text requests preserve candidate IDs/text/source spans, map every overlapping gold span, and reject unmappable tasks.
- [x] 1.3 Add pair/model projection tests proving Dense-FT selects candidates by variant with identical graph-free sampling settings and distinct typed pair identity.
- [x] 1.4 Add retrieval tests proving frozen provenance-unit Dense equals provenance-path's initial Dense ordering without graph input.
- [x] 1.5 Add checkpoint metadata tests proving flat and provenance-unit Dense-FT checkpoints load only for their exact requested variant, with legacy metadata remaining flat.
- [x] 1.6 Add workflow-plan tests proving frozen provenance-unit Dense is test-only and provenance-unit Dense-FT schedules text-only pair/train/retrieve/evaluate stages without graph or frozen-R-GCN work.

## 2. Add typed variant configuration

- [x] 2.1 Add a closed `flat|provenance_unit` variant with flat defaults to Dense and Dense-FT method configs.
- [x] 2.2 Reject provenance-unit variants outside ISETrace while preserving every existing flat composition.
- [x] 2.3 Extend pair-build config, normalized config, final results, inspection, tracking, and artifact/cache inputs with the resolved variant.
- [x] 2.4 Verify Hydra commands override only `method.variant` and reuse all canonical Dense/Dense-FT profile settings.

## 3. Implement provenance-unit text supervision

- [x] 3.1 Generalize the ISETrace exact-span Dense adapter behind explicit flat/provenance-unit entry points without changing flat behavior.
- [x] 3.2 Make the provenance-unit entry point emit graph-free task-local `TextRankingRequest` and dependency-free `EvidenceLabel` values over provenance candidates.
- [x] 3.3 Route ISETrace pair construction by Dense-FT variant and preserve the variant in pair artifact/cache inputs.
- [x] 3.4 Route Dense-FT train/dev projection by variant while retaining trajectory group IDs and task-local Recall@5 selection.

## 4. Enforce retrieval and checkpoint variant identity

- [x] 4.1 Select provenance candidates for the frozen Dense variant while reusing the unchanged Dense scorer and tie-breaking.
- [x] 4.2 Extend Dense-FT run/checkpoint metadata with the closed candidate-view variant and flat-compatible default.
- [x] 4.3 Require exact checkpoint variant equality in Dense-FT retrieval before loading/ranking.
- [x] 4.4 Select provenance candidates for Dense-FT retrieval while preserving existing method labels and complete source-backed rankings.

## 5. Wire the existing workflow lifecycle

- [x] 5.1 Keep the non-training Dense workflow branch and resolve only test preparation plus the shared encoder for provenance-unit Dense.
- [x] 5.2 Keep the Dense-FT workflow branch and pass the resolved variant through pair planning, model training, checkpoint resolution, and artifact collection.
- [x] 5.3 Load provenance graphs only for path/R-GCN methods; schedule no graph or frozen-R-GCN stage for either Dense provenance-unit variant.
- [x] 5.4 Bump only affected pair/train/rank implementation identities and verify flat/provenance-unit artifacts cannot alias.
- [x] 5.5 Add a synthetic CPU behavior smoke covering frozen retrieval and the fine-tuned lifecycle boundary with exact-span evaluation.

## 6. Update maintained documentation

- [x] 6.1 Update architecture and retrieval-contract documentation with the Dense candidate-view variant and graph-free boundary without increasing the public method count.
- [x] 6.2 Update ISETrace non-training and Dense-FT runbooks with copy-pastable variant commands, matched settings, seeds, outputs, and interpretation limits.
- [x] 6.3 Update command/catalog documentation so users can inspect and run both controls without reintroducing template supervision or graph claims.

## 7. Verify the implementation

- [x] 7.1 Run focused config, ISETrace adapter, pair, Dense-FT metadata/training, workflow, retrieval, evaluation, and output tests.
- [x] 7.2 Run the full pytest suite, Ruff, BasedPyright, compileall, and `git diff --check` under the repository environment.
- [x] 7.3 Run `openspec validate add-isetrace-provenance-unit-dense-baselines --strict` and review the final diff for graph leakage, duplicated model/trainer paths, accidental method-ID expansion, and unsupported scientific claims.

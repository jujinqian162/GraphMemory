## 1. Freeze ISETrace Dense-FT contracts with failing tests

- [x] 1.1 Add Registry/config tests requiring `dense_ft` to support ISETrace execution provenance while retaining `TextRankingRequest`, model-directory artifacts, no graph input, and unchanged evidence-family support.
- [x] 1.2 Add flat supervision tests proving natural and template exact spans map to every overlapping flat chunk, preserve task IDs/source spans, and fail clearly when a task has no positive chunk.
- [x] 1.3 Add pair-stage tests proving ISETrace `dense_ft` uses flat requests and graphless labels while `provenance_rgcn` retains provenance candidates and candidate-neighbor edges.
- [x] 1.4 Add workflow-plan tests proving ISETrace Dense-FT schedules prepare train/dev/test, text-only pairs, train, flat retrieve, span evaluate, and aggregate without EvidenceGraph or frozen R-GCN encoding stages.
- [x] 1.5 Add effective-config/tracking tests proving ISETrace graph-neighbor negatives are zero in pair/train scientific identity while evidence Dense-FT configuration is unchanged.

## 2. Add neutral exact-span and flat supervision adapters

- [x] 2.1 Extract exact `SourceSpan` overlap into a neutral reusable utility and migrate provenance R-GCN mapping without behavior change.
- [x] 2.2 Implement an ISETrace flat training adapter that converts typed ranking/label records into flat `TextRankingRequest` values and candidate-level `EvidenceLabel` positives.
- [x] 2.3 Validate duplicate/alignment/unknown/unmappable task cases before pair construction or training.
- [x] 2.4 Export the new adapter through the ISETrace package boundary without exposing template/provenance authoring details to Dense-FT.

## 3. Make pair construction method-native

- [x] 3.1 Add trainable method identity to the typed pair-build configuration and all callers/artifact identities.
- [x] 3.2 Dispatch ISETrace pair tasks to the flat adapter for `dense_ft` and the existing provenance adapter for `provenance_rgcn`.
- [x] 3.3 Keep evidence dataset pair projection and EvidenceGraph sampling behavior unchanged.
- [x] 3.4 Add one family-aware effective Dense-FT config projection that sets only ISETrace graph-neighbor negatives to zero and is reused by workflow planning, pair construction, and tracking.

## 4. Wire the ISETrace Dense-FT lifecycle

- [x] 4.1 Widen Registry compatibility for `dense_ft` and preserve strict flat payload/model-directory validation.
- [x] 4.2 Make Dense-FT workflow preparation pass the canonical ISETrace trajectory source for train, dev, and test.
- [x] 4.3 Project ISETrace train/dev artifacts through the flat adapter when constructing `DenseFinetuneTrainPayload`; retain existing evidence-dataset projection.
- [x] 4.4 Reuse the current Dense-FT retrieve builder over ISETrace flat candidates and preserve complete rankings plus exact source spans.
- [x] 4.5 Add cache/resume/artifact tests proving method-native pairs and the effective sampling policy participate in scientific identity.

## 5. Add task-local origin-aware dev selection

- [x] 5.1 Add typed Dense-FT task-local evaluator inputs carrying requests, candidate-level labels, and natural/template origin outside model text/features.
- [x] 5.2 Implement batched normalized query/passage encoding and per-task cosine ranking with Recall@2/5/10 and MRR grouped by origin.
- [x] 5.3 Select mixed-dev checkpoints by `dev_natural_recall_at_5` and template-only dev checkpoints by `dev_template_recall_at_5`.
- [x] 5.4 Record selection origin, selected metric, epoch-0/epoch-end values, best epoch, and best value in training metrics and model artifact metadata.
- [x] 5.5 Preserve the existing evidence Dense-FT `InformationRetrievalEvaluator` and `eval_dev_cos_sim_map@100` behavior.

## 6. Prevent trajectory-local in-batch false negatives

- [x] 6.1 Carry optional trajectory group identity in Dense-FT training examples without serializing it into anchor/positive/negative model texts.
- [x] 6.2 Implement a deterministic seed-controlled batch sampler that emits at most one example per trajectory in each ISETrace batch.
- [x] 6.3 Integrate grouped batches with the existing `InputExample`, `MultipleNegativesRankingLoss`, and `SentenceTransformer.fit()` path; retain ordinary shuffled loading for evidence datasets.
- [x] 6.4 Add tests for repeated queries, multiple positive chunks, template/natural examples on one trajectory, deterministic replay, partial batches, and loader-length/global-step accounting.

## 7. End-to-end behavior and documentation

- [x] 7.1 Add a synthetic CPU workflow smoke proving pair, model-directory, metadata, training metrics, predictions, exact-span metrics, and aggregate artifacts are produced with no graph stage.
- [x] 7.2 Verify the natural-only ISETrace test boundary and existing Recall/MRR/Coverage/Full Support/span-F1/evidence-density outputs, with path/edge metrics unavailable.
- [x] 7.3 Update maintained Registry/retrieval contracts, architecture, config examples, and ISETrace runbook for Dense-FT graph boundaries and effective text-only negatives.
- [x] 7.4 Document natural-only and natural:template=1:2 commands over the same split seed/test set, plus seeds 13/17/29 and the engineering-pilot-before-formal-run rule.

## 8. Verification

- [x] 8.1 Run focused ISETrace adapter, pair, Dense-FT data/training, Registry, workflow, retrieval, and evaluation tests.
- [x] 8.2 Run the repository broad pytest gate, Ruff, BasedPyright, compileall, and `git diff --check` in the effective host environment.
- [x] 8.3 Run `openspec validate add-isetrace-dense-ft --strict` and review the final diff for graph leakage, duplicate trainer/scorer paths, evidence Dense-FT regressions, and unsupported path/edge claims.
- [x] 8.4 Stop after verified engineering smoke behavior; do not add paper result tables before formal reviewed-data seeds complete.

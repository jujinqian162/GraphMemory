## Context

See `proposal.md` for motivation. The current branch already provides the two scientific inputs required by this baseline: ISETrace preparation emits one shared lossless flat-chunk view with exact `SourceSpan` coordinates, and the trajectory-grouped query composition emits natural/template train and dev plus natural-only test. It also provides a maintained Dense-FT implementation that builds train rows from candidate-level pairs, trains SentenceTransformers 2.7 with `MultipleNegativesRankingLoss`, selects a model-directory checkpoint, and reuses the ordinary Dense retriever.

The remaining boundaries do not align. Registry still restricts `dense_ft` to evidence retrieval; ISETrace pair preparation always compiles the provenance R-GCN view; Dense-FT training asks the dataset for node-ID evidence labels that ISETrace intentionally does not expose; the workflow may schedule an EvidenceGraph because the evidence profile enables graph-neighbor negatives; and the current IR evaluator uses one cross-task corpus rather than each query's trajectory-local candidate set.

ISETrace flat candidate IDs are local to a prepared ranking task and exact evaluation is span-based. A natural or template gold span may overlap multiple chunks because flat chunking uses overlap. Those chunks are all valid positives. The same trajectory may also contribute multiple natural queries, templates, positive chunks, and explicit hard-negative rows, which makes ordinary shuffled in-batch-negative training susceptible to treating semantically positive texts from the same trajectory as negatives.

## Goals / Non-Goals

**Goals:**

- Reuse the existing public `dense_ft` method, model, checkpoint, and Dense inference implementation.
- Train and retrieve only over ISETrace `flat_candidates`, preserving exact source coordinates.
- Compile both natural and template exact spans into flat candidate positives through one neutral rule.
- Keep pair planning and training projection explicit about the selected method-native ISETrace view.
- Match dev checkpoint selection to task-local final retrieval and the RQ2 primary metric, natural Recall@5.
- Keep same-trajectory examples out of the same in-batch-negative batch deterministically.
- Preserve all existing evidence-dataset Dense-FT and provenance R-GCN behavior.

**Non-Goals:**

- No graph input, graph-derived feature, native trace, motif feature, or graph-neighbor negative for Dense-FT.
- No new retrieval method ID, alternate dense scorer, cross-encoder, graph-seeded Dense-FT, or Dense-FT-seeded provenance R-GCN.
- No new trainer framework, dependency, DDP/Accelerate integration, copied optimizer loop, or second SentenceTransformers API path.
- No change to ISETrace chunking, query ownership, natural/template selection, test labels, or exact-span evaluation.
- No hyperparameter sweep framework or paper result in this engineering change.

## Decisions

### 1. Reuse `dense_ft` and widen its family support

Registry will declare the existing `dense_ft` definition compatible with both evidence retrieval and execution provenance. It retains `TextRankingRequest`, a model-directory artifact, no graph requirement, and no native-edge capability. Method identity stays `dense_ft` in checkpoints, predictions, aggregates, and MLflow.

Alternative considered: add `isetrace_dense_ft`. Rejected because training, checkpoint, and ranking semantics are still ordinary Dense-FT; a second ID would duplicate configuration and imply a provenance-native model.

### 2. Compile exact spans to flat positives in the ISETrace dataset adapter

A neutral exact-span-overlap helper will own the rule currently embedded privately in the provenance R-GCN adapter. An ISETrace flat training adapter will:

1. project each ranking record to `TextRankingRequest(flat_candidates)`;
2. mark every candidate having at least one exact source-span overlap with the label as positive;
3. emit an `EvidenceLabel` containing those flat candidate IDs and no dependency edges;
4. fail with the task ID if no positive is found or any positive lies outside the request.

Natural and template records take this same path. Template preparation has already converted focused output content into exact source spans, so Dense-FT never reads `TemplateSupervisionRecord`, motif participants, or the provenance graph.

Alternative considered: map template focus IDs directly through provenance ownership and then into flat text. Rejected because it creates a second label path and leaks graph-specific authoring semantics into a flat baseline.

### 3. Make pair projection method-native rather than dataset-only

`PairBuildConfig` will carry the trainable method identity used to select the pair projection. For ISETrace:

- `dense_ft` invokes the flat adapter and creates graphless `TrainPairBuildTask` values;
- `provenance_rgcn` retains the current provenance adapter and candidate-neighbor edges.

Evidence datasets retain their current text/evidence graph path. The method identity participates in pair artifact/cache identity, preventing a flat pair artifact and provenance pair artifact over the same prepared data from colliding.

Alternative considered: infer the view from whether an EvidenceGraph argument is present. Rejected because both ISETrace methods omit EvidenceGraph while requiring different candidates and labels.

### 4. Derive one family-aware effective Dense-FT sampling config

A config-layer helper will return the configured Dense-FT method unchanged for evidence datasets and a copy with `hard_graph_neighbor_per_positive=0` for ISETrace. Workflow planning will use that one effective method for EvidenceGraph scheduling, pair config, train-stage scientific config/tracking, and artifact identity.

Easy-random, BM25-hard, Dense-hard, pool size, and training settings remain unchanged. ISETrace full defaults may enable Dense-hard negatives independently, but graph-neighbor sampling is never accepted.

Alternative considered: change `configs/stage/dense_ft.yaml` globally. Rejected because it would silently alter completed HotpotQA, 2Wiki, and MuSiQue baselines. A manual CLI override was also rejected because a registered method/dataset composition must be runnable and scientifically explicit by default.

### 5. Use a typed Dense-FT training projection for ISETrace

`materialize_dense_finetune_model()` will dispatch only its dataset projection:

- evidence datasets continue to use the existing dataset text requests and evidence labels;
- ISETrace loads typed ranking/label records and uses the flat training adapter for train and dev.

The resulting `DenseFinetuneTrainPayload` remains unchanged: requests, candidate-level labels, materialized pairs, output directory, and model directory. This keeps dataset semantics out of the Dense-FT model package.

Alternative considered: teach `evidence_labels_for_dataset()` to convert ISETrace globally. Rejected because that function promises evidence-node labels and could accidentally make graph/evidence workflows accept span-only ISETrace data.

### 6. Replace ISETrace global-corpus dev evaluation with task-local evaluation

The existing evidence Dense-FT evaluator remains unchanged. ISETrace uses a task-local evaluator payload containing each query, its candidate texts, relevant local IDs, and query origin. One evaluation call batch-encodes deduplicated query/passages, then computes normalized cosine scores and ranks only within each request.

Per origin it reports at least Recall@2, Recall@5, Recall@10, and MRR. The selected metric is:

- `dev_natural_recall_at_5` when any natural dev query exists;
- `dev_template_recall_at_5` for template-only dev.

The epoch-0 and epoch-end metric tracker continues to own best-model saving. Training artifact metadata records `selection_query_origin`, `selected_metric_name`, and selected value. Evidence datasets continue using `eval_dev_cos_sim_map@100` and the current `InformationRetrievalEvaluator`.

Alternative considered: keep the global corpus evaluator and only rename its score. Rejected because cross-trajectory candidates are absent from final retrieval, so that score measures a different task and can select a different checkpoint.

### 7. Add a trajectory-grouped batch sampler without replacing the loss

Dense-FT training examples will carry an optional `group_id`; for ISETrace it is the ranking record's `graph_id`, and for evidence datasets it remains absent. A deterministic batch sampler will shuffle group queues and examples from `random_seed`, then emit batches containing at most one example per group. When all examples have no group ID, the current shuffled `DataLoader` path is retained unchanged.

The sampler changes only batch composition. Rows remain two- or three-text `InputExample` values, explicit hard negatives remain supported, and `MultipleNegativesRankingLoss` plus `SentenceTransformer.fit()` remain authoritative.

Alternative considered: replace MNRL with a custom multi-positive contrastive loss. Rejected for the first version because it would change optimization semantics, require a copied or parallel trainer path, and make comparison with established Dense-FT less direct.

### 8. Reuse flat retrieval and exact-span evaluation unchanged

The retrieve stage already selects flat ISETrace candidates for `dense_ft`, and the Dense-FT builder already loads strict model-directory metadata. After Registry/workflow support is widened, test retrieval can reuse this path. Ranked nodes retain candidate source spans, so the existing ISETrace span suite produces Recall, MRR, Coverage@512/1024/2048, Full Support, span F1, and evidence density. No path or edge label is synthesized from provenance input.

### 9. Keep the experiment protocol explicit but small

The maintained runbook documents primary configurations over deterministic, trajectory-count-based train/dev/test splits. Selected natural trajectories contribute all available authored queries, while selected template trajectories contribute one template query each.

- natural-only train/dev;
- size-matched natural:template `1:2`, matching the current provenance R-GCN supervision budget.

Formal model seeds are 13, 17, and 29 after one seed-13 engineering pilot fixes learning rate and epoch count. Independent seeds can be scheduled across visible GPUs by the existing run orchestration; single-model distributed training is outside this change.

## Risks / Trade-offs

- [Span overlap and chunk overlap can create multiple positives per query] -> Treat all overlapping chunks as valid positives and use trajectory-safe batches so they do not become same-batch negatives.
- [Trajectory grouping can reduce batch packing efficiency] -> Use deterministic round-robin group queues; report actual loader length and accept partial final batches rather than violating the false-negative boundary.
- [Template supervision is coarser than natural spans] -> Consume only the exact spans already materialized by preparation and evaluate/select on natural dev whenever available.
- [Task-local evaluation can be expensive each epoch] -> Deduplicate passage encodings within the dev split and batch encoder calls while preserving per-task ranking.
- [Effective pair configuration can diverge across stages] -> Derive it once before workflow planning and pass the same typed value into pair/train identity and tracking.
- [Existing Dense-FT tests assume the global evaluator] -> Keep that path as the evidence-family default and add ISETrace-specific evaluator tests rather than changing completed evidence baselines.

## Migration Plan

1. Add failing Registry, config, pair-view, span-mapping, workflow-plan, evaluator, and batch-sampler tests.
2. Extract neutral exact-span overlap and implement the ISETrace flat supervision adapter.
3. Add method-aware pair projection and family-aware effective Dense-FT sampling, then wire ISETrace train/dev/test planning without EvidenceGraph stages.
4. Add the task-local origin-aware evaluator and metadata while preserving evidence Dense-FT evaluation.
5. Add deterministic trajectory-grouped batching and verify the existing SentenceTransformers fit path.
6. Run a synthetic CPU end-to-end workflow, maintained quality gates, and strict OpenSpec validation before any server pilot.

Rollback removes ISETrace from Dense-FT Registry compatibility and the method-aware flat pair/train projection. Existing ISETrace prepared artifacts, evidence Dense-FT checkpoints, and provenance R-GCN checkpoints remain valid because no persisted dataset or checkpoint schema is replaced.

## Open Questions

None blocking. Learning rate and maximum epoch count are operational pilot choices; they do not change the contracts or implementation architecture.

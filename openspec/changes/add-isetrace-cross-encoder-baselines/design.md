## Context

ISETrace preparation already persists flat chunks and provenance content units with exact source spans. Existing pair construction maps those views to task-local positives and graph-free random/BM25 negatives. The missing scorer is a query-candidate interaction model that can serve as the strongest flat comparison and, in the provenance-unit variant, a candidate-matched graph-free control.

## Goals / Non-Goals

**Goals:**

- Add one closed Cross-Encoder method with flat/provenance-unit candidate views.
- Train and rank the complete task-local candidate set without graph input or a first-stage retrieval bottleneck.
- Match variants on backbone, loss, negative sampling, optimizer, epochs, seed, and dev selection.
- Persist strict scientific and checkpoint identity.
- Reuse existing prepared candidate views and pair artifacts.

**Non-Goals:**

- No graph features, traversal, R-GCN seeding, or provenance path trace.
- No top-N tuning on the test split.
- No change to ISETrace data, evaluation, Dense-FT, or residual R-GCN.
- No multiple-model search or test-set model selection.

## Decisions

### 1. Use a distinct `cross_encoder` method with a candidate-view variant

Unlike provenance-unit Dense, the Cross-Encoder changes the scoring architecture and training objective, so it receives a real method ID. `variant=flat|provenance_unit` changes only the candidate view. Both variants are ISETrace-only in this study.

### 2. Fine-tune a fixed pretrained reranker backbone

The registered `BAAI/bge-reranker-base` checkpoint is pinned to Hugging Face revision `2cfc18c9415c912f9d8155881c133215df768a70` and loaded through SentenceTransformers 2.7 `CrossEncoder`. It already contains a pretrained one-logit sequence-classification reranking head; the experiment MUST NOT replace it with a random head. Query and candidate text form the tokenizer pair. Training uses `BCEWithLogitsLoss`, one epoch, AdamW, fixed seed, and task-local Recall@5 checkpoint selection over the pretrained epoch-0 model and each completed epoch.

### 3. Score every local candidate

Each task has a bounded local trajectory candidate set, so inference directly scores all query-candidate pairs. This removes Dense candidate-pool recall as a confound and makes the reported candidate pool exact and reproducible. Batching controls runtime only.

### 4. Reuse graph-free sampled pair artifacts

The existing pair builder is generalized to accept `cross_encoder`. Exact-span positives and configured random/BM25 negatives are unchanged. Cross-Encoder training materializes one labelled pair per selected positive or negative candidate, with no graph-neighbor negatives on ISETrace.

### 5. Select by task-local development Recall@5

The pretrained epoch-0 model and every completed epoch score each dev task's complete candidate set. Checkpoint selection uses macro development Recall@5, matching the current ISETrace trainable lifecycle's ranking objective.

### 6. Fail closed on checkpoint identity

Metadata records method, variant, base model, max length, train/eval batch sizes, selection rule, and device. Retrieval rejects a checkpoint with a different method or variant before model loading. Artifact and Prefect identities additionally include prepared, pair, development, and immutable backbone source digests.

## Risks / Trade-offs

- **External reranker drift** → pin `BAAI/bge-reranker-base` to revision `2cfc18c...`, content-address the downloaded directory, and record that identity in every model artifact.
- **Full-candidate scoring is slower** → batch all pairs per task and report measured online latency; avoid hidden top-N recall constraints.
- **Long pairs are truncated** → freeze and record max length 512, matching the backbone limit and existing candidate construction.
- **Pointwise sampling may underuse negatives** → consume every persisted positive and sampled negative row; keep both variants' sampling identical.

## Migration Plan

1. Add config, metadata, scorer, workflow, and graph-isolation tests.
2. Implement Cross-Encoder contracts/training/retrieval and input provisioning.
3. Wire existing pair, model, rank, evaluation, output, and tracking stages.
4. Add Hydra/profile configuration and operations documentation.
5. Run smoke and full quality gates before remote seeds 13/17/29.

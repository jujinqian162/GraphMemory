## Purpose

Defines strong graph-free Cross-Encoder scorers over matched ISETrace flat and provenance-unit candidate views.

## ADDED Requirements

### Requirement: Cross-Encoder exposes matched candidate views
The system SHALL expose `cross_encoder variant=flat|provenance_unit`, SHALL default to flat, and SHALL reject the method outside ISETrace. Both variants MUST share backbone, tokenizer, max length, loss, sampling, optimizer, epochs, random seed, and checkpoint-selection rule.

#### Scenario: Compose matched variants
- **WHEN** flat and provenance-unit Cross-Encoder jobs use the same profile and seed
- **THEN** only the selected candidate view differs
- **AND** both variants have graph-neighbor sampling disabled

### Requirement: Cross-Encoder remains graph-free
The method SHALL receive only task-local text ranking requests and MUST NOT receive a provenance graph, edge, graph identifier feature, query-type feature, or native provenance trace.

#### Scenario: Rank provenance units
- **WHEN** the provenance-unit variant ranks an ISETrace task
- **THEN** it scores the exact persisted provenance candidates and source spans
- **AND** no graph artifact is loaded or supplied

### Requirement: Training uses exact-span labels and persisted sampled pairs
The trainer SHALL load the fixed pretrained `BAAI/bge-reranker-base` one-logit reranking head and MUST NOT replace it with a randomly initialized classifier. It SHALL convert every positive and negative row in the selected candidate-view pair artifact into a labelled query-candidate example. It SHALL optimize one-logit binary cross entropy and SHALL fail when pair tasks, candidate IDs, or labels do not align.

#### Scenario: Build labelled examples
- **WHEN** one task has exact-span positive candidates and sampled negatives
- **THEN** every persisted pair becomes one labelled Cross-Encoder example
- **AND** no candidate from another task is introduced

### Requirement: Development selection uses complete local rankings
The trainer SHALL score every candidate in every development task before computing macro Recall@5. It SHALL evaluate the pretrained epoch-0 reranker and every completed epoch and SHALL persist the checkpoint with the best development Recall@5. Epoch 0 is valid only because the fixed backbone contains a pretrained reranking head.

#### Scenario: Select a checkpoint
- **WHEN** an epoch improves macro development Recall@5
- **THEN** that epoch's model is saved as the selected checkpoint
- **AND** training metrics record the epoch, global step, loss, current Recall@5, and best Recall@5

### Requirement: Inference scores the complete candidate set
The retriever SHALL score all task-local query-candidate pairs directly and return each candidate exactly once, ordered by descending score with candidate ID as the deterministic tie-breaker. It MUST NOT depend on a Dense top-N pool.

#### Scenario: Run the strongest flat baseline
- **WHEN** a flat ISETrace task contains N candidates
- **THEN** exactly N Cross-Encoder scores are produced
- **AND** the ranking is evaluated under the existing six token budgets and Budget-AUC contract

### Requirement: Checkpoint identity fails closed
Checkpoint metadata SHALL record method, variant, backbone, max length, batch settings, and selection rule. Retrieval MUST reject method or variant mismatches before loading weights.

#### Scenario: Reject a candidate-view mismatch
- **WHEN** a flat Cross-Encoder run receives provenance-unit checkpoint metadata
- **THEN** retrieval fails with an explicit variant mismatch before scoring

### Requirement: Outputs preserve trainable-run reproducibility
Pair, model, prediction, evaluation, run, and tracking artifacts SHALL record method `cross_encoder`, candidate-view variant, training seed, and scientific input digests. Formal results SHALL use seeds 13, 17, and 29 over the fixed ISETrace test artifact.

#### Scenario: Complete a formal run
- **WHEN** one Cross-Encoder seed finishes
- **THEN** the run contains training metrics, a strict model origin, complete ranked prefixes, per-task metrics, six-budget aggregates, Budget-AUC, and measured retrieval latency

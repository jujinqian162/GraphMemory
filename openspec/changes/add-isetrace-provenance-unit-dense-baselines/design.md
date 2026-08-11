## Context

See `proposal.md` for motivation. ISETrace preparation already persists two source-backed candidate views per task: lossless flat chunks and provenance content units. `text_ranking_requests_for_dataset()` can project either view, `provenance_path` already initializes traversal with the ordinary frozen Dense ranker over provenance units, and ISETrace Dense-FT already provides exact-span supervision, task-local development evaluation, trajectory-grouped batches, and a graph-free training lifecycle.

The missing control is not a new model or retrieval method. It is an explicit candidate-view variant on the existing Dense methods. Current Dense always selects flat candidates; path/R-GCN methods select provenance candidates; and Dense-FT pair/train/checkpoint contracts do not yet record a candidate view.

## Goals / Non-Goals

**Goals:**

- Add a closed flat/provenance-unit variant to Dense and Dense-FT while preserving flat defaults.
- Reuse the existing Dense scorer and Dense-FT trainer without copying model or optimization code.
- Make the selected candidate view explicit at every projection boundary and in scientific identity.
- Keep both provenance-unit controls graph-free and preserve exact source spans through evaluation.
- Make flat/provenance-unit Dense-FT checkpoints fail closed when exchanged.
- Preserve all existing method and evidence-dataset behavior.

**Non-Goals:**

- No new public retrieval method ID.
- No graph traversal, graph convolution, relation feature, provenance path, or graph-neighbor negative.
- No new encoder, loss, optimizer, trainer API, dependency, evaluation metric, or candidate construction.
- No Dense-FT-seeded provenance R-GCN, cross-encoder baseline, template supervision, split change, token-budget expansion, or paper result generation.

## Decisions

### 1. Model candidate representation as a Dense variant

`DenseMethodConfig` and `DenseFinetuneMethodConfig` gain `variant: Literal["flat", "provenance_unit"] = "flat"`. Commands use `method=dense method.variant=provenance_unit` and `method=dense_ft method.variant=provenance_unit`. Existing configs and checkpoints retain flat semantics by default.

This matches the scientific intervention: encoder, scorer, loss, optimizer, and workflow remain the same while only candidate segmentation changes. It also follows the repository's existing method-plus-variant pattern for R-GCN ablations and avoids expanding the closed method enum and static dispatch for non-algorithms.

Alternative: add `provenance_unit_dense` and `provenance_unit_dense_ft` method IDs. Rejected after review because it makes a candidate-view control look like a new algorithm and requires unnecessary enum/config/registry/workflow branches.

### 2. Validate provenance-unit variants as ISETrace-only

Resolved-config validation rejects `variant=provenance_unit` unless the dataset family is execution provenance. Flat remains valid on all existing Dense/Dense-FT datasets. The variant is exposed by `ResolvedExperimentConfig.variant` and therefore enters normalized config, final results, tracking, and run delivery through existing mechanisms.

Alternative: make the variant silently fall back to flat on evidence datasets. Rejected because it would produce mislabeled experiments.

### 3. Keep provenance-unit variants on `TextRankingRequest`

ISETrace task projection selects `provenance_candidates` when either Dense method resolves to the provenance-unit variant and produces graph-free text requests. Frozen retrieval calls the existing `DenseTaskRetriever`; fine-tuned retrieval loads the existing SentenceTransformer directory and calls the same ranker. Provenance graphs remain unloaded.

The per-task retrieval method stays `dense` or `dense_ft`; run-level result/tracking records additionally carry `variant=provenance_unit`, as existing R-GCN variants do. No retrieval result schema expansion is needed.

Alternative: route through `ExecutionProvenanceRankingRequest` and ignore its graph. Rejected because carrying a graph weakens the no-graph control.

### 4. Generalize exact-span Dense supervision by candidate view

The ISETrace adapter owns one internal exact-span-to-text-candidate routine parameterized by flat or provenance view, with explicit public entry points. Both validate task/label alignment, map every overlapping candidate, reject empty positives, and emit dependency-free `EvidenceLabel` values. The provenance R-GCN adapter remains separate because it produces graph requests and graph-neighbor context.

Alternative: derive labels through the provenance R-GCN adapter and discard graph fields. Rejected because graph-model assumptions must not define a graph-free baseline.

### 5. Persist variant in pair and checkpoint identity

`PairBuildConfig` gains the resolved Dense candidate-view variant. Workflow construction passes `method.variant`, so flat and provenance-unit pair tasks differ in typed Prefect/cache input even when all sampling hyperparameters match.

Dense-FT run config and model metadata gain the same variant with default `flat`. Training writes it; retrieval requires exact equality with the requested config before loading/ranking. Existing metadata without the field parses as flat and remains valid only for flat Dense-FT.

Alternative: infer the view from run names or checkpoint paths. Rejected because operational paths are not reliable scientific identity.

### 6. Reuse closed workflow branches

The existing non-training Dense branch handles both frozen variants; the existing Dense-FT branch handles both trainable variants. Only candidate projection branches on `method.variant`. For provenance-unit Dense-FT, preparation schedules train/dev/test, text-only pairs, shared training, test retrieval, span evaluation, and output projection. Neither variant schedules graph construction or frozen R-GCN embeddings.

No runtime plugin registry or new capability abstraction is introduced.

### 7. Preserve matched scientific settings

Dense-FT's existing ISETrace effective configuration already forces graph-neighbor negatives to zero. Both variants use that same effective config. Hydra uses the same method file and profile values; only `method.variant` differs. Tests compare encoder, pair sampling, data, trainer, selection, seed, and token-budget settings between variants.

Existing prepared ISETrace artifacts remain reusable because both candidate views are already persisted.

## Risks / Trade-offs

- **[Per-task predictions contain method but not variant]** → Run-level config, final result, tracking, manifest, pair artifacts, and checkpoint metadata record the variant; document aggregation labels as method+variant, consistent with R-GCN ablations.
- **[A missed cache input could alias candidate views]** → Put variant directly in typed pair/model/retrieve method config arguments and test differing artifact/cache identities.
- **[A graph can leak through generic provenance helpers]** → Use only `TextRankingRequest`, do not load provenance graphs for Dense variants, and add workflow/builder tests that fail on graph access.
- **[Legacy checkpoint metadata lacks variant]** → Default it to flat and strictly reject it for provenance-unit retrieval.
- **[Multiple provenance candidates can overlap one gold span]** → Retain every overlap exactly as flat Dense-FT does and preserve trajectory-grouped batching.

## Migration Plan

1. Add contract tests for variant composition, dataset validation, candidate projection, graph isolation, pair identity, checkpoint identity, workflow planning, and retrieval behavior.
2. Add the variant fields to Dense/Dense-FT and pair/checkpoint contracts with flat defaults.
3. Generalize ISETrace Dense supervision and pair/model/retrieve projections by explicit view.
4. Wire variant validation and existing workflow branches without new method IDs.
5. Update maintained architecture and ISETrace operations documentation.
6. Run focused tests, a synthetic CPU lifecycle smoke where practical, full quality gates, and strict OpenSpec validation.

Rollback removes the variant fields/branches. Existing flat configs and checkpoints remain valid because flat is the default; provenance-unit artifacts can be discarded independently.

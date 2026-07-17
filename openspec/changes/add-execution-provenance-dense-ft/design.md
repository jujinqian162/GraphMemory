## Context

`twowiki_provenance` owns both a native `ExecutionProvenanceRankingRequest` projection and a flat `TextRankingRequest` projection over the same ToolOutput candidates. BM25 and Dense already use the flat projection, but Registry limits `dense_ft` to evidence retrieval. The workflow also decides whether pair building needs EvidenceGraph from the method-config class: every Dense-FT pair stage is therefore wired to an EvidenceGraph path even when the selected dataset family cannot produce one.

Dense-FT itself is graph-free. It trains a Sentence Transformers model from question/candidate text and materialized positive/negative pairs, saves a model directory plus strict metadata, then returns a flat ranked-node result. The existing trainer, checkpoint format, retrieval builder, evaluation request, and aggregate row are already compatible with execution-provenance candidate IDs.

## Goals / Non-Goals

**Goals:**

- Run the existing public `dense_ft` method end to end on `twowiki_provenance`.
- Keep request projection dataset-owned and keep Dense-FT graph-free.
- Preserve evidence-dataset Dense-FT negative sampling and result behavior.
- Make pair-stage graph requirements follow the selected task family and effective sampling contract rather than a concrete method-config class.
- Prove the real plan and a smoke workflow produce Dense-FT pairs, a model directory, predictions, metrics, and aggregate output without EvidenceGraph stages.

**Non-Goals:**

- Adding a Dense-FT-seeded provenance R-GCN or changing either existing provenance retriever.
- Feeding `ExecutionProvenanceGraph` topology, bindings, or native traces into Dense-FT.
- Changing generated provenance data, labels, split policy, or checkpoint formats.
- Changing Dense-FT sampling defaults for HotpotQA, standard 2Wiki, or MuSiQue.

## Decisions

### 1. Reuse `dense_ft` and widen only its supported task family

Registry will declare `dense_ft` compatible with both evidence retrieval and execution provenance while retaining `TextRankingRequest`, no required graph artifact, the Dense-Finetune lifecycle, and the existing `best_model` directory contract. Results remain under the same method ID so tables compare one algorithm across datasets rather than inventing a dataset-prefixed alias.

Alternative considered: add `execution_provenance_dense_ft`. Rejected because its request, trainer, model, and ranking semantics would be identical to `dense_ft`; a second ID would duplicate configuration and falsely imply a provenance-native model.

### 2. Keep provenance projection flat and ranking-side only

Training and retrieval will consume `TwoWikiProvenanceToTextRankingRequest`, whose candidate IDs are ToolOutput IDs and whose text contains only title/text ranking inputs. Gold evidence remains label-side. Dense-FT will not receive `ExecutionProvenanceGraph`, native relations, bindings, answer text, or gold dependency edges.

Alternative considered: convert the provenance graph into an EvidenceGraph so the existing pair path remains unchanged. Rejected because Dense-FT does not consume graph structure and the synthetic conversion would violate the repository's typed graph boundary.

### 3. Derive pair graph usage from the selected dataset family

The planner will identify the run's retrieval family once. Evidence-family trainable methods retain the current EvidenceGraph dependency and configured graph-neighbor negatives. Execution-provenance trainable methods receive no EvidenceGraph input. For `dense_ft` on that family, the emitted pair-stage config will copy the configured sampling policy but set `hard_graph_neighbor_per_positive` to `0`; the generated stage YAML and pair summary therefore expose the effective value.

The pair script will build one generic `TrainPairBuildTask` per text request and attach a graph only when one was supplied. The existing pair builder remains the enforcement point that rejects a positive graph-neighbor count when all tasks omit graphs. This removes the method-name special case while preserving strict validation.

Alternative considered: set Dense-FT graph-neighbor negatives to zero globally. Rejected because it would silently change established evidence-dataset experiments. Requiring a manual CLI override was also rejected because selecting a Registry-supported method/dataset pair should produce a runnable plan by default.

### 4. Reuse the existing Dense-FT lifecycle unchanged after pairs

The ordinary `pairs -> train -> retrieve -> evaluate -> aggregate` stages and existing model directory paths remain authoritative. Retrieval already constructs a `FlatRetrievalBuildPayload` with the dataset task family, so widening Registry compatibility is sufficient at runtime. Dense-FT remains a flat method and does not claim native-edge traces or provenance path reasoning.

## Risks / Trade-offs

- [Provenance Dense-FT sees a different negative mix from evidence Dense-FT] -> Record the effective zero graph-neighbor count in stage config and pair summary, while retaining easy/BM25/dense negatives.
- [Family logic becomes duplicated] -> Use one planner helper for dataset-to-family resolution and reuse it for compatibility and pair-stage decisions.
- [A flat method is mistaken for provenance reasoning] -> Keep its request type, capabilities, docs, and output trace graph-free and describe it as a supervised text baseline.
- [Synthetic label-derived data encourages overclaiming] -> Preserve the runbook disclosure and make no graph-reasoning claim from Dense-FT performance.

## Migration Plan

1. Add failing Registry and workflow-plan tests for `dense_ft` on `twowiki_provenance`.
2. Widen Registry family support and generalize pair planning/construction.
3. Update the runbook and run focused tests plus a one-example CPU smoke workflow.

Rollback removes the family widening and pair-planner adaptation; existing evidence Dense-FT artifacts and all provenance data/checkpoints remain valid.

## Open Questions

None blocking. Full-run batch sizing and result interpretation remain operational experiment choices rather than implementation contracts.

## Why

The current `twowiki_provenance` R-GCN ablations do not cleanly isolate the intended signals: most gold dependency edges enter through semantic fallback and are then strongly attenuated by `1 / rank`, uniform-weight ablation changes total message mass, and generic hard negatives add substantial training volume without measurable node-ranking benefit. The resulting small Full Support reversals are dominated by a saturated test metric and checkpoint noise even though the full model remains materially better on head ranking and path/edge recall.

This change makes graph construction, weighted message passing, hard-negative supervision, structured inference, and evaluation agree on one auditable contract. Its success criterion is a fair and reproducible comparison whose metric movements have a mechanism-level explanation; it does not require every ablation cell to be lower than the full model.

## What Changes

- Replace fallback-as-top-k-repair with an explicit hidden gold spine plus structurally matched semantic branches. Every source keeps fixed visible out-degree and receives the same head/tail branch policy, while the gold dependency remains label-conditioned structure without a gold-only field, relation, degree, rank band, or weight convention.
- Version the source-aware successor policy and generate one semantic-head proposal plus one deterministic rank-banded branch proposal per source. Preserve BM25, dense, and hybrid strategies as explicit construction interventions, with source-local confidence calibration and complete rank/fallback audits.
- Replace raw `1 / semantic_rank` edge weights with bounded, source-mass-preserving feed weights. Non-feed structural edges remain weight `1.0`; feed weights retain a positive structural floor and have a fixed mean within each source successor group.
- Redefine `wo_edge_weight` as a mass-matched removal of within-group confidence rather than setting every edge to `1.0`, so the ablation preserves endpoints, relation IDs, degree, and total message mass.
- Add provenance-native hard negatives from competing logical successors/predecessors, deduplicate candidates across negative sources with an explicit hardness precedence, and use task-balanced pairwise ranking supervision. `wo_hard_negatives` retains only easy random negatives and invalidates from the pair stage.
- Make learned logical-edge scores participate in bounded top-M structured candidate selection and add a truthful `wo_edge_rerank` control. Edge inference may abstain below a configured threshold instead of emitting one successor for every selected source.
- Select checkpoints with a declared joint dev objective over Full Support@5, MRR, and Edge F1@10, while logging every component and retaining deterministic tie-breaking.
- Add Edge Precision@10 and Edge F1@10 to provenance evaluation, deliver graph-construction diagnostics with every generated dataset, and define a paired multi-seed ablation protocol using the full available dev split.
- Add leakage and fairness controls for matched branch-rank distributions, weight-mass preservation, shuffled logical endpoints, topology-free inference, pair-source deduplication, and identical data/model provenance across compared variants.
- **BREAKING**: bump the generated `twowiki_provenance` raw schema and provenance R-GCN checkpoint schema. Existing generated raw/prepared datasets, pair artifacts, checkpoints, predictions, and metrics are diagnostic-only and must not be reused by the new comparison.

## Capabilities

### New Capabilities

- `provenance-graph-calibration`: hidden-gold-spine construction, matched semantic branches, bounded source-mass-preserving weights, versioned scorer identity, and leakage/fallback distribution audits.
- `provenance-rgcn-structured-learning`: mass-matched weight consumption and ablation, provenance-native hard negatives, task-balanced ranking loss, edge-aware structured candidate selection, abstaining logical-edge inference, and self-describing checkpoints.
- `provenance-rgcn-evaluation-validity`: edge precision/F1, joint checkpoint selection, paired multi-seed experiment controls, provenance equality checks, and acceptance criteria that do not encode a desired leaderboard ordering.

### Modified Capabilities

None. The repository currently has no promoted baseline specs under `openspec/specs/`; this change establishes current contracts without treating completed change-local specs as active modification targets.

## Impact

- Dataset conversion, scoring, records, validation, projectors, runbook, smoke fixture, generated manifests, and statistics under `graph_memory/datasets/twowiki_provenance/`, `scripts/data/`, `configs/dataset/`, and `docs/40-operations/`.
- Provenance R-GCN config, tensorization, loss, training, inference, checkpoint loading, method registration, ablation registration, and pair-stage routing under `graph_memory/models/provenance_rgcn/`, `graph_memory/training_pairs/`, `graph_memory/stages/`, `graph_memory/experiment/`, and `graph_memory/registry/`.
- Evaluation contracts/tables and result deliveries for provenance-capable methods, including new edge precision/F1 columns and construction-audit artifacts.
- Focused dataset/model/evaluation tests plus full-profile, same-digest, multi-seed experiment jobs. No answer generation, live-agent execution capture, joint dense-encoder finetuning, or changes to standard `twowiki` semantics are included.

# 2Wiki synthetic execution-provenance runbook

`twowiki_provenance` is a separately named synthetic retrieval benchmark. It is not a claim that 2Wiki contains agent executions. A one-time deterministic converter maps each recoverable ordered two-evidence chain to one ordinary typed output dependency and adds structurally matched non-gold branches. Standard `twowiki` files and results are unchanged.

## Audit and convert

The default sources are the labeled local 2Wiki train/dev files. Source train becomes target train; source dev is split deterministically into target dev/test.

```powershell
uv run python scripts/data/convert_2wiki_to_execution_provenance.py --audit-only

uv run python scripts/data/convert_2wiki_to_execution_provenance.py `
  --train-source data/2wiki/raw/train.json `
  --dev-source data/2wiki/raw/dev.json `
  --output-dir data/twowiki_provenance/v3/raw `
  --candidate-cap 32 --seed 13 --dev-fraction 0.5 `
  --edge-scorer hybrid --successors-per-output 2 `
  --hybrid-dense-weight 0.5 --semantic-temperature 0.1 `
  --weight-floor 0.5 `
  --near-rank-bucket 2:4 --mid-rank-bucket 5:8 --tail-rank-bucket '9:*'
```

`--edge-scorer` accepts `bm25`, `dense`, or `hybrid`; BM25-only and dense-only are construction interventions, while formal v3 data uses the pinned hybrid scorer. Dense requests for all sources in one graph are batched. Every source ranks all non-self candidates from `question + source evidence`, receives one rank-1 semantic head and one deterministic near/mid/tail branch, and has exactly two `feeds` edges.

The ordered gold dependency is materialized as an ordinary head or branch edge. When it occupies a branch bucket, an ordinary non-gold branch from the same bucket is required; otherwise the record is rejected as `unmatched_gold_branch_bucket`. No gold/fallback/support field enters ranking input. This is intentionally label-conditioned synthetic construction: it tests retrieval from a hidden required path plus matched distractors, not whether 2Wiki itself contains real agent trajectories.

Each source-local confidence is temperature-normalized and converted to `weight = 0.5 + 0.5 * probability`; therefore two feed weights always sum to `1.5`. `wo_edge_weight` replaces the two values by their source mean (`0.75/0.75`) rather than raising both to `1.0`. `manifest.json` records all scorer/query/bucket/calibration and encoder identities. `statistics.json` records gold/non-gold bucket, rank, weight, head-rate, rejection, and source-mass audits. Every feed edge has the same confidence metadata schema; gold diagnostics remain label-side.

Before switching `configs/dataset/twowiki_provenance.yaml`, generate the pilot twice and compare all outputs:

```powershell
Get-FileHash data/twowiki_provenance/v3/pilot-a/* | Sort-Object Path
Get-FileHash data/twowiki_provenance/v3/pilot-b/* | Sort-Object Path
```

The raw, manifest, and statistics hashes must match. Confirm zero source-mass violations, fixed out-degree two, no ranking-side forbidden fields, and matching non-gold buckets for every gold branch. Then set the three dataset sources to `data/twowiki_provenance/v3/raw/*.json` and set capacities from the actual manifest counts; do not reuse historical v2 capacities blindly.

## Smoke and comparison runs

Start with one train/dev/test record:

```powershell
uv run python experiment/run.py -m `
  name=twowiki_provenance_smoke dataset=twowiki_provenance profile=smoke device=cpu `
  method=bm25,dense,dense_ft,graphrag,execution_provenance_retriever,execution_provenance_rgcn_retriever
```

`dense_ft` is the supervised flat-text baseline. It trains and ranks the same dataset-owned ToolOutput candidates used by BM25 and Dense, but it does not receive the execution-provenance graph, bindings, or dependency labels. Its pair stage keeps easy/BM25/dense negatives, sets graph-neighbor negatives to zero, and schedules no EvidenceGraph stage. It therefore does not demonstrate provenance reasoning even if it improves candidate recall.

The trainable provenance method follows `prepare -> pairs -> train -> rank -> evaluate` and does not build an EvidenceGraph artifact. Its typed pair task carries text, the execution graph, and the label. Full sampling uses successor/predecessor/dense/BM25/easy counts `2/1/1/1/2` per positive, deduplicates by candidate with provenance-first precedence, and trains task-balanced pairwise logistic candidate loss plus class-balanced edge BCE. Schema-v3 checkpoints store construction, pair, loss, weight, inference, selection, variant, and best-component identities; v2 provenance and evidence checkpoints are rejected.

## Provenance R-GCN ablations

`execution_provenance_rgcn_retriever` exposes five public ablations in addition to `full_rgcn`:

- `wo_graph`: set message-passing layers to zero.
- `wo_edge_type`: share one message transform across all relation IDs.
- `wo_edge_weight`: remove within-source confidence while preserving endpoints, relations, directions, and source feed mass.
- `wo_hard_negatives`: rebuild pairs without successor, predecessor, BM25, or dense hard negatives while retaining easy random negatives and the full edge loss.
- `wo_edge_rerank`: reuse full pairs/checkpoint and return raw candidate-logit order while retaining edge diagnostics.

Run each variant as an independent job. For two GPUs, for example:

```powershell
uv run python experiment/run.py `
  name=twowiki_provenance_rgcn_full dataset=twowiki_provenance profile=quick device=cuda:0 `
  method=execution_provenance_rgcn_retriever method.variant=full_rgcn

uv run python experiment/run.py `
  name=twowiki_provenance_rgcn_wo_graph dataset=twowiki_provenance profile=quick device=cuda:1 `
  method=execution_provenance_rgcn_retriever method.variant=wo_graph
```

The jobs are peer Prefect/MLflow runs. Model-only variants reuse compatible pair artifacts; `wo_hard_negatives` owns variant-specific pairs and every downstream asset. Evidence-graph-only variants such as `wo_bridge` and `wo_seed_score` are intentionally unsupported for this method because the provenance R-GCN does not consume those signals.

Formal runs use the complete generated dev split through `profile=provenance_full`; the historical 500-example full-profile dev subset is not valid for v3 checkpoint selection. The selected objective is `0.50 * Full Support@5 + 0.25 * MRR + 0.25 * Edge F1@10`, with ties resolved by Full Support@5, Edge F1@10, MRR, then earlier epoch. Freeze construction, thresholds, loss weights, and selection from train/dev before reading test results.

```powershell
$variants = @('full_rgcn','wo_graph','wo_edge_type','wo_edge_weight','wo_hard_negatives','wo_edge_rerank')
foreach ($seed in 13,17,29) {
  foreach ($variant in $variants) {
    uv run python experiment/run.py `
      name="twowiki_provenance_v3_${variant}_s${seed}" `
      dataset=twowiki_provenance profile=provenance_full device=cuda `
      seed=$seed method=execution_provenance_rgcn_retriever method.variant=$variant
  }
}
```

Report every seed, mean/std, query-paired confidence intervals, discordant Full Support counts, Edge Precision/Recall/F1@10, average retrieved edges, abstention, and exact artifact identities. Use `scripts/analyze_provenance_ablation.py` on the collected per-seed/per-task JSON. A valid ablation may beat full on a metric; correctness gates test fairness and invariants, not a desired ordering.

For a frozen-method diagnostic before training R-GCN:

```powershell
uv run python experiment/run.py `
  name=twowiki_provenance_frozen_quick dataset=twowiki_provenance profile=quick device=cuda `
  method=execution_provenance_retriever
```

## Interpretation

Report Recall/Evidence F1/Full Support for ToolOutput candidates and path metrics for contracted output dependencies. Edge Precision guards against recall-through-overproduction; abstention reports how often considered sources emit no accepted dependency. Always disclose that topology is label-derived and synthetic. Method ordering is an experimental result, not a converter invariant; topology-free and explicitly requested shuffled-feed diagnostics are required before claiming graph reasoning gains.

Rollback means pointing the dataset config back to the old v2 raw directory and restoring the previous method config. Never mix v2/v3 raw, prepared, pair, checkpoint, prediction, or evaluation rows: schemas and artifact digests are intentionally incompatible, and there is no translation fallback. Do not update paper-facing result claims until the complete three-seed v3 matrix exists.

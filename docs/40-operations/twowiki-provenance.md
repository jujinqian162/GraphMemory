# 2Wiki synthetic execution-provenance runbook

`twowiki_provenance` is a separately named synthetic retrieval benchmark. It is not a claim that 2Wiki contains agent executions. A one-time deterministic converter maps each recoverable ordered two-evidence chain to one ordinary typed output dependency and adds structurally matched non-gold branches. Standard `twowiki` files and results are unchanged.

## Automatic in-flow transform

Conversion is a Prefect-cached `transform_twowiki_task` inside the experiment flow, running before `prepare_split_task`. There is no standalone convert script. The dataset config points `splits.*.source` at the labeled local 2Wiki files (train source becomes target train; dev source is split deterministically into target dev/test), and the `transform` block in `configs/dataset/twowiki_provenance.yaml` holds all conversion parameters. Formal v3 uses the pinned `hybrid` scorer.

Running the flow with `dataset=twowiki_provenance` transforms automatically:

```powershell
uv run python experiment/run.py `
  name=twowiki_provenance_full dataset=twowiki_provenance profile=provenance_full device=cuda `
  method=execution_provenance_rgcn_retriever
```

The transform writes `data/twowiki_provenance/raw/<version_tag>/{train,dev,test}.json`, where `version_tag = v{schema_version}-{digest}` and the digest is a canonical-JSON sha256 over the schema version, the transform parameters, and (for dense/hybrid) the content-addressed encoder digest. Different schema versions or parameters land in different version directories and never overwrite each other. No `manifest.json` or `statistics.json` is emitted; parameters and the encoder identity are implicit in the version tag and the Prefect artifact origin.

`edge_scorer` accepts `bm25`, `dense`, or `hybrid`; BM25-only and dense-only are construction interventions, while formal v3 data uses the pinned hybrid scorer. When `edge_scorer` is `dense` or `hybrid`, the dense model is resolved through `resolve_encoder_source` and its content digest participates in both the Prefect cache key and the version tag, so changing model weights (even without renaming) forces a re-transform. Every source ranks all non-self candidates from `question + source evidence`, receives one rank-1 semantic head and one deterministic near/mid/tail branch, and has exactly two `feeds` edges.

The ordered gold dependency is materialized as an ordinary head or branch edge. When it occupies a branch bucket, an ordinary non-gold branch from the same bucket is required; otherwise the record is rejected as `unmatched_gold_branch_bucket`. No gold/fallback/support field enters ranking input. This is intentionally label-conditioned synthetic construction: it tests retrieval from a hidden required path plus matched distractors, not whether 2Wiki itself contains real agent trajectories.

Each source-local confidence is temperature-normalized and converted to `weight = 0.5 + 0.5 * probability`; therefore two feed weights always sum to `1.5`. `wo_edge_weight` replaces the two values by their source mean (`0.75/0.75`) rather than raising both to `1.0`. Every feed edge has the same confidence metadata schema; gold diagnostics remain label-side.

Split counts do not need to be pinned in the config. `splits.*.capacity` is optional, and profiles that request `all_available` consume every valid record the transform produces (after `strict=False` drops unrecoverable records). Set a `capacity` only when you deliberately want to cap a split; `all_available` without a capacity reads whatever the transformed file contains.

## Smoke and comparison runs

Start with one train/dev/test record:

```powershell
uv run python experiment/run.py -m `
  name=twowiki_provenance_smoke dataset=twowiki_provenance profile=smoke device=cpu `
  method=bm25,dense,dense_ft,graphrag,execution_provenance_retriever,execution_provenance_rgcn_retriever
```

`dense_ft` is the supervised flat-text baseline. It trains and ranks the same dataset-owned ToolOutput candidates used by BM25 and Dense, but it does not receive the execution-provenance graph, bindings, or dependency labels. Its pair stage keeps easy/BM25/dense negatives, sets graph-neighbor negatives to zero, and schedules no EvidenceGraph stage. It therefore does not demonstrate provenance reasoning even if it improves candidate recall.

The trainable provenance method follows `prepare -> pairs -> train -> rank -> evaluate` and does not build an EvidenceGraph artifact. Its typed pair task carries text, the execution graph, and the label. Full sampling uses successor/predecessor/dense/BM25/easy counts `2/1/1/1/2` per positive, deduplicates by candidate with provenance-first precedence, and trains task-balanced pairwise logistic candidate loss plus class-balanced edge BCE. Checkpoint schema v4 stores `candidate_loss_protocol=provenance-candidate-loss-v2`, disconnected-union batching, construction, pair, weight, inference, selection, variant, and best-component identities. Legacy provenance schema-v3 single-graph checkpoints and evidence schema-v2 checkpoints are rejected rather than translated.

## Physical graph batching

Evidence and provenance R-GCN now use seeded map-style PyTorch DataLoaders over precomputed CPU task tensors. The encoder runs during one-time materialization, not in `__getitem__`, a worker, or epoch-time collation; the initial worker policy is `num_workers=0` and incomplete final batches are retained.

`per_device_graph_batch_size` is the number of task graphs in one disconnected-union forward, backward pass, and optimizer step. The full provenance profile uses a true graph batch of `8`, quick uses `8`, smoke uses `1`, and full evidence uses `128`. The incomplete final DataLoader batch is retained and normalized by its actual task or supervised-sample count. Every report must state the configured graph batch and observed short-tail task count.

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

Formal runs use the complete generated dev split through `profile=provenance_full`; the historical 500-example full-profile dev subset is not valid for dataset-v3 formal checkpoint selection under checkpoint schema v4. The selected objective is `0.50 * Full Support@5 + 0.25 * MRR + 0.25 * Edge F1@10`, with ties resolved by Full Support@5, Edge F1@10, MRR, then earlier epoch. Freeze construction, thresholds, loss weights, and selection from train/dev before reading test results.

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

## Non-trained EPGM presets

`execution_provenance_retriever` is one implementation with two frozen presets
selected by `method.variant`: `typed_beam` (default, reported) and
`dependency_path` (schema-gated restriction, ablation only). Run them as peer
jobs when the ablation row is needed:

```powershell
uv run python experiment/run.py `
  name=twowiki_provenance_epgm_typed_beam dataset=twowiki_provenance profile=quick device=cuda:0 `
  method=execution_provenance_retriever method.variant=typed_beam

uv run python experiment/run.py `
  name=twowiki_provenance_epgm_dependency_path dataset=twowiki_provenance profile=quick device=cuda:1 `
  method=execution_provenance_retriever method.variant=dependency_path
```

The preset participates in the Prefect cache key and is tagged as
`graph_memory.variant`, so the two rows are independently cached and
attributable. See
[`docs/40-operations/stateless-graph-retrieval.md`](stateless-graph-retrieval.md)
for the per-axis definition of each preset.

## Interpretation

Report Recall/Evidence F1/Full Support for ToolOutput candidates and path metrics for contracted output dependencies. Edge Precision guards against recall-through-overproduction; abstention reports how often considered sources emit no accepted dependency. Always disclose that topology is label-derived and synthetic. Method ordering is an experimental result, not a converter invariant; topology-free and explicitly requested shuffled-feed diagnostics are required before claiming graph reasoning gains.

Rollback means switching back to the older code branch and config; the schema version and transform parameters revert with them, the version tag reproduces, and the flow hits the Prefect cache for the old version directory (if it is still on disk) without a manual config edit or a re-transform. Version directories accumulate under `data/twowiki_provenance/raw/`; keeping old ones on disk is what makes rollback a cache hit. Deleting a version directory is safe but means rolling back to it will re-run the transform instead of hitting the cache. Never mix raw, prepared, pair, checkpoint, prediction, or evaluation rows across schema versions: schemas and artifact digests are intentionally incompatible, and there is no translation fallback. Do not update paper-facing result claims until the complete three-seed dataset-v3/checkpoint-v4 matrix exists.

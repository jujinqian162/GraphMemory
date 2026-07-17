# 2Wiki synthetic execution-provenance runbook

`twowiki_provenance` is a separately named synthetic retrieval benchmark. It is not a claim that 2Wiki contains agent executions. A one-time deterministic converter maps each recoverable ordered two-evidence chain to one ordinary typed output dependency and adds structurally matched non-gold branches. Standard `twowiki` files and results are unchanged.

## Audit and convert

The default sources are the labeled local 2Wiki train/dev files. Source train becomes target train; source dev is split deterministically into target dev/test.

```powershell
uv run python scripts/data/convert_2wiki_to_execution_provenance.py --audit-only

uv run python scripts/data/convert_2wiki_to_execution_provenance.py `
  --train-source data/2wiki/raw/train.json `
  --dev-source data/2wiki/raw/dev.json `
  --output-dir data/twowiki_provenance/raw `
  --candidate-cap 32 --seed 13 --dev-fraction 0.5 `
  --edge-scorer bm25 --successors-per-output 2
```

`--edge-scorer` accepts `bm25`, `dense`, or `hybrid`. Dense and hybrid construction use the configured dense encoder; hybrid combines BM25 and dense rank positions with `--hybrid-dense-weight`. For every source output, the query is the task question plus that source title/text, and only the top semantic successors become `feeds` edges. If the gold successor falls outside top-k, the converter replaces the last proposal and records the original semantic rank plus a fallback flag; it never fills branches by random topology.

For the repository's pinned local 2Wiki files, the audit accepts 75,793 train records and 6,084 source-dev records; the latter becomes 3,042 dev and 3,042 test records. If source hashes differ, use `manifest.json` counts to update the dataset capacities before a full-profile run.

Generated `manifest.json` records source hashes, schema/seed/split parameters, graph scorer settings, and rejection counts. `statistics.json` records candidate, node, edge, fixed `feeds` out-degree, gold semantic-rank, and gold-fallback summaries. Every generated `feeds` edge also carries its semantic scorer, rank, score, and rank-derived weight. Full generated raw data is intentionally not committed.

## Smoke and comparison runs

Start with one train/dev/test record:

```powershell
uv run python experiment/run.py -m `
  name=twowiki_provenance_smoke dataset=twowiki_provenance profile=smoke device=cpu `
  method=bm25,dense,dense_ft,graphrag,execution_provenance_retriever,execution_provenance_rgcn_retriever
```

`dense_ft` is the supervised flat-text baseline. It trains and ranks the same dataset-owned ToolOutput candidates used by BM25 and Dense, but it does not receive the execution-provenance graph, bindings, or dependency labels. Its pair stage keeps easy/BM25/dense negatives, sets graph-neighbor negatives to zero, and schedules no EvidenceGraph stage. It therefore does not demonstrate provenance reasoning even if it improves candidate recall.

The trainable provenance method follows `prepare -> pairs -> train -> rank -> evaluate` and does not build an EvidenceGraph artifact. The pair Task materializes configured easy/BM25/dense negatives, and candidate BCE consumes only that artifact. The model uses relation-specific graph convolution over all typed nodes plus binding-schema-aware `feeds` relations, then independently scores legal output transitions. It has no beam, dynamic oracle, maximum-step, or path-loss contract. Its schema-v2 checkpoint family is distinct from both evidence R-GCN methods and rejects earlier provenance checkpoints.

## Provenance R-GCN ablations

`execution_provenance_rgcn_retriever` exposes four executable variants:

- `wo_graph`: set message-passing layers to zero.
- `wo_edge_type`: share one message transform across all relation IDs.
- `wo_edge_weight`: replace graph artifact weights with uniform `1.0` weights.
- `wo_hard_negatives`: rebuild pairs without BM25, dense, or graph-neighbor hard negatives while retaining easy random negatives.

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

For a frozen-method diagnostic before training R-GCN:

```powershell
uv run python experiment/run.py `
  name=twowiki_provenance_frozen_quick dataset=twowiki_provenance profile=quick device=cuda `
  method=execution_provenance_retriever
```

## Interpretation

Report Recall/Evidence F1/Full Support for ToolOutput candidates and path metrics for contracted output dependencies. Always disclose that topology is label-derived and synthetic. Method ordering is an experimental result, not a converter invariant; topology-only and shuffled-edge controls are required before claiming graph reasoning gains.

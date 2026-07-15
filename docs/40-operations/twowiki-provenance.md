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
uv run python experiment/plan.py `
  name=twowiki_provenance_smoke dataset=twowiki_provenance profile=smoke device=cpu `
  'methods=[bm25,dense,graphrag,execution_provenance_retriever,execution_provenance_rgcn_retriever]'

uv run python experiment/run.py `
  name=twowiki_provenance_smoke dataset=twowiki_provenance profile=smoke device=cpu `
  'methods=[bm25,dense,graphrag,execution_provenance_retriever,execution_provenance_rgcn_retriever]'
```

The trainable provenance method reuses the existing `pairs -> train -> retrieve -> evaluate -> aggregate` workflow phases but does not schedule an EvidenceGraph stage. The pair stage materializes configured easy/BM25/dense negatives, and candidate BCE consumes only that artifact. The model uses relation-specific graph convolution over all typed nodes plus binding-schema-aware `feeds` relations, then independently scores legal output transitions. It has no beam, dynamic oracle, maximum-step, or path-loss contract. Its schema-v2 checkpoint family is distinct from both evidence R-GCN methods and rejects earlier provenance checkpoints.

For a frozen-method diagnostic before training R-GCN:

```powershell
uv run python experiment/run.py `
  name=twowiki_provenance_frozen_quick dataset=twowiki_provenance profile=quick device=cuda `
  'methods=[bm25,dense,graphrag,execution_provenance_retriever]'
```

## Interpretation

Report Recall/Evidence F1/Full Support for ToolOutput candidates and path metrics for contracted output dependencies. Always disclose that topology is label-derived and synthetic. Method ordering is an experimental result, not a converter invariant; topology-only and shuffled-edge controls are required before claiming graph reasoning gains.

# 2Wiki synthetic provenance runbook

`twowiki_provenance` is a synthetic retrieval benchmark built from recoverable 2Wiki two-evidence chains. It is not a claim that 2Wiki contains agent executions. Standard `twowiki` results are independent.

## In-flow transform

`transform_twowiki_task` runs inside the experiment Flow before prepare. Config: `configs/dataset/twowiki_provenance.yaml`. Formal data uses the pinned hybrid edge scorer.

```powershell
uv run python experiment/run.py `
  name=twowiki_provenance_full dataset=twowiki_provenance profile=provenance_full device=cuda `
  method=execution_provenance_rgcn_retriever
```

Raw output: `data/twowiki_provenance/raw/<version_tag>/{train,dev,test}.json` with `version_tag = v{schema_version}-{digest}`. Ranking input has no gold/support fields. Each source has equal public `feeds` degree and matched distractor branches.

## Methods on this dataset

| Method | Notes |
|---|---|
| BM25 / Dense / Dense-FT / GraphRAG | Flat or entity baselines; Dense-FT does not see the provenance graph |
| `execution_provenance_retriever` | EPGM; default `ppr_steiner` |
| `execution_provenance_rgcn_retriever` | Trainable; prepare → pairs → train → rank → eval; no EvidenceGraph stage |

Provenance R-GCN ablations (one job each): `full_rgcn`, `wo_graph`, `wo_edge_type`, `wo_edge_weight`, `wo_hard_negatives`, `wo_edge_rerank`.

```powershell
foreach ($seed in 13,17,29) {
  foreach ($variant in @('full_rgcn','wo_graph','wo_edge_type','wo_edge_weight','wo_hard_negatives','wo_edge_rerank')) {
    uv run python experiment/run.py `
      name="twowiki_provenance_v3_${variant}_s${seed}" `
      dataset=twowiki_provenance profile=provenance_full device=cuda `
      seed=$seed method=execution_provenance_rgcn_retriever method.variant=$variant
  }
}
```

Checkpoint selection (formal): `0.50 * Full Support@5 + 0.25 * MRR + 0.25 * Edge F1@10`. Aggregate with `scripts/analyze_provenance_ablation.py`. Report mean±std over seeds `{13,17,29}`; disclose synthetic topology.

## EPGM defaults

```powershell
uv run python experiment/run.py `
  name=twowiki_provenance_epgm dataset=twowiki_provenance `
  profile=provenance_full device=cuda:0 `
  method=execution_provenance_retriever
# diagnostics: method.variant=typed_beam | dependency_path
```

See [`stateless-graph-retrieval.md`](stateless-graph-retrieval.md) for algorithm notes.

## Rollback

Switch code/config branch; version tag + Prefect cache hit the old raw directory if it still exists. Never mix prepared/pair/checkpoint rows across schema versions.

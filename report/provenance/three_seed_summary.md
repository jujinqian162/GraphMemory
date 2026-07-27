# RQ2 provenance three-seed summary (v2)

Source runs: `runs/prgcn-twp-*-v2` (rsynced from autodl-graphmemory).

Seeds: `{13, 17, 29}`

## Completed variants

| Variant | FS@5 μ±σ | MRR μ±σ | Path@10 μ±σ | EdgeP@10 μ±σ | EdgeF1@10 μ±σ | ΔFS@5 [95% CI] | ΔEdgeF1 (seed range) |
|---|---|---|---|---|---|---|---|
| full_rgcn | 97.08±0.21 | 95.13±0.10 | 91.32±0.39 | 84.19±0.65 | 87.61±0.41 | -- | -- |
| wo_graph | 84.51±0.60 | 83.33±0.91 | 78.53±0.39 | 56.78±1.33 | 65.88±0.86 | 12.57 [11.77, 13.32] | 21.73 [21.44, 22.29] |
| wo_edge_type | 95.12±1.13 | 92.60±0.45 | 88.85±2.17 | 75.75±3.08 | 81.69±1.12 | 1.95 [1.54, 2.37] | 5.92 [5.41, 6.77] |
| wo_edge_weight | 96.48±0.28 | 94.97±0.49 | 89.54±1.05 | 80.95±0.30 | 85.00±0.37 | 0.60 [0.23, 0.96] | 2.61 [2.37, 2.81] |
| wo_edge_rerank | 96.64±0.17 | 95.09±0.11 | 91.32±0.39 | 84.21±0.63 | 87.62±0.40 | 0.43 [0.27, 0.60] | -0.01 [-0.01, 0.00] |

## Pending

- `wo_hard_negatives` multiseed

## Artifacts

- `report/provenance/ablation_rows.json`
- `report/provenance/ablation_analysis.json`
- `report/provenance/seed_tables.json`
- `report/provenance/edgef1_seed_deltas.json`

## Notes

- ΔFS@5: query-paired bootstrap over matched seed-query pairs (`n=8553`, 2000 resamples, seed 13).
- ΔEdgeF1: seed-level micro-average differences (Edge F1 is micro-aggregated, not in `per_task.jsonl`).
- Paper updates: `docs/raw/paper.tex`, `docs/raw/exp.tex`.

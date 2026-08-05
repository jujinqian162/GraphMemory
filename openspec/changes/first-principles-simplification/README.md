# first-principles-simplification

Remove orchestration, registry, payload, result, and configuration duplication while preserving all current scientific methods and observable experiment behavior.

## Closeout Evidence

### Validation

The final branch state passed:

- `uv run pytest -q`: 193 passed.
- `uv run ruff check .`: passed.
- `uv run basedpyright --level error`: 0 errors, 0 warnings, 0 notes.
- `python -m compileall -q graph_memory experiment scripts tests`: passed.
- `openspec validate first-principles-simplification --strict`: passed.
- `git diff --check`: passed.

All eight method configs composed under `profile=smoke` with their supported domain: BM25, Dense, Dense-FT, GraphRAG, Dense R-GCN, Dense-FT R-GCN, provenance path, and provenance R-GCN. Every resolved trainable split had count 1; bounded ISETrace train/dev/test also each resolved to count 1.

### Representative Smoke Workflows

| Family | Command identity | Result |
|---|---|---|
| Flat | `final_flat_bm25_smoke` | HotpotQA, 1 test task, Recall@10 1.0 |
| Dense-FT | `final_dense_ft_smoke` | HotpotQA, 1 train/dev/test task, Recall@10 1.0 |
| Evidence graph | `final_evidence_rgcn_smoke` | HotpotQA full R-GCN, 1 train/dev/test task, completed |
| Provenance graph | `final_provenance_rgcn_smoke` | ISETrace full R-GCN, 1 train/dev/test task, completed |

Repeating `final_flat_bm25_smoke` returned all three Prefect tasks from cache. The asset manifest, final metrics, per-task rows, and failure-case output retained identical SHA-256 hashes across the rerun. Existing behavior tests continue to cover deterministic preparation, graph construction, rankings, metrics, manifests, Prefect cache identity/resume, and MLflow tags/metrics/assets.

### Deletion Scan

Repository scans found no remaining definitions or imports for the deleted result envelopes/batches, dynamic retrieval registry/spec, execution task, score pipeline, trainable graph adapter, retrieval settings/build payloads, graph scoring factory, universal prepared DTO, benchmark replay DTOs, stage result DTOs, motif catalog, template selector, conversion DTOs, retired stage trainer/payload modules, or `combined.json` authority.

### Net Reduction

Compared with baseline commit `2258c25`:

| Measure | Baseline | Final | Change |
|---|---:|---:|---:|
| Production Python lines (`graph_memory`) | 23,621 | 21,890 | -1,731 |
| Test Python lines | 7,896 | 7,863 | -33 |
| Docs/config lines | 1,947 | 1,939 | -8 |
| Production Python modules | 175 | 160 | -15 |
| Production classes | 353 | 281 | -72 |

Across review range `8a1234f..HEAD`, the final diff is 4,103 insertions and 6,321 deletions, a net reduction of 2,218 lines.

### Residual Risks

- Bounded ISETrace preparation materializes one task but still scans the complete source to establish deterministic trajectory ownership; the provenance smoke spent roughly 90-100 seconds per split in preparation.
- ISETrace natural queries remain unreviewed engineering inputs, so smoke metrics are validation evidence rather than paper claims.
- Concrete artifact-ref subclasses remain because parameterized generic aliases do not preserve basedpyright `isinstance` narrowing.
- Prefect and MLflow remain operational dependencies; their cache/resume and tracking boundaries are retained intentionally.

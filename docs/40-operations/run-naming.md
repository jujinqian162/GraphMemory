# Run naming convention

`name=<...>` is the stable human-facing identity shared by `runs/<name>/`, the Prefect flow, and the MLflow run. It does not enter scientific cache identity, but automation uses it to discover cohorts. Names must therefore be concise, deterministic, and parseable.

## Canonical order

```text
<dataset>[_<data-version>]_<study-tag>_<method>[_<variant>][_s<seed>][_<scope>]
```

Fields always appear in that order:

1. **dataset** — benchmark identity;
2. **data version** — only when multiple frozen/generated revisions coexist;
3. **study tag** — the experiment cohort or research question;
4. **method** — retrieval/training family;
5. **variant** — only when the method has multiple scientific variants;
6. **seed** — only for stochastic/trainable runs;
7. **scope** — only for non-formal subsets such as smoke or quick runs.

Examples:

```text
isetrace_v7_main_bm25
isetrace_v7_main_dft_pu_s13
isetrace_v7_e2rel_rgcn_full_s13
isetrace_v7_e2rel_rgcn_homog_s17
isetrace_v7_e2rel_rgcn_randedge_s29
hotpotqa_main_dft_rgcn_full_s13
musique_ablation_rgcn_nograph_s29
isetrace_v7_e2rel_rgcn_full_s13_smoke
```

A multi-seed study changes only the `s<seed>` field. A control study changes only the variant field. This makes cohort discovery safe, for example:

```bash
--name-prefix isetrace_v7_e2rel_
```

## Required information

| Field | Required | Rule |
|---|---:|---|
| dataset | always | Use the registered dataset slug: `hotpotqa`, `twowiki`, `musique`, `isetrace`. |
| data-version | when revisions coexist | Use a short immutable revision such as `v7`; do not write `latest`, `new`, or `final`. |
| study-tag | always | Short scientific cohort identifier such as `main`, `e2rel`, `e3sup`, `ablation`, `pilot`, or `audit`. Reuse the same tag across every method and seed in one comparison. |
| method | always | Use one canonical abbreviation from the table below. |
| variant | when applicable | Describe the scientific variant, not runtime placement. |
| seed | trainable/stochastic runs | Write `s13`, `s17`, `s29`; deterministic methods omit it. `split_seed` is fixed benchmark metadata and is not repeated in the name. |
| scope | non-formal runs only | Append `smoke` or `quick`. Formal/full runs omit `full`. |

## Canonical abbreviations

### Methods

| Configuration | Name token |
|---|---|
| `bm25` | `bm25` |
| frozen `dense` | `dense` |
| `dense_ft` | `dft` |
| `cross_encoder` | `ce` |
| `graphrag` | `graphrag` |
| `provenance_path` | `ppath` |
| provenance R-GCN | `rgcn` |
| evidence Dense R-GCN | `dense_rgcn` |
| evidence Dense-FT R-GCN | `dft_rgcn` |

### Common variants

| Configuration variant | Name token |
|---|---|
| `flat` | `flat` |
| `provenance_unit` | `pu` |
| `full_rgcn` | `full` |
| `wo_graph` | `nograph` |
| `homogeneous_gcn` | `homog` |
| `wo_feeds` | `nofeeds` |
| `wo_execution_ownership` | `noexec` |
| `wo_artifact_io` | `noio` |
| `wo_chunk_adjacency` | `nochunk` |
| `random_edges` | `randedge` |
| `wo_hard_negatives` | `nohardneg` |

Add a new abbreviation here before using it broadly. Do not invent two tokens for the same method or variant within one project.

## Syntax and exclusions

- Use lowercase ASCII snake case: `[a-z0-9]+(?:_[a-z0-9]+)*`.
- Keep the name informative but short; do not repeat information already implied by a token.
- **Never include timestamps or dates.** Use Git history and Prefect/MLflow timestamps for chronology.
- Do not include GPU/device placement, host name, worker count, cache flags, Hydra job number, Prefect ID, or MLflow ID.
- Do not use subjective lifecycle words such as `new`, `latest`, `final`, `fixed`, `rerun`, or `try2`.
- If scientific inputs intentionally change, assign a meaningful study tag or immutable data version rather than an ordinal retry suffix.
- A name must identify exactly one Hydra job. Do not reuse an existing name unless intentionally resuming/reconstructing that same job.

## Reporting workflow

First aggregate statistically comparable runs with `scripts/aggregate_main_results.py`. Then render the machine-readable JSON without ad-hoc formatting code:

```bash
uv run python scripts/deliver/report_experiment_results.py \
  --input results/isetrace/e2rel.json \
  --name-prefix isetrace_v7_e2rel_ \
  --metrics 'Recall@5,MRR,Full Support@2048 Tokens' \
  --title 'ISETrace E2 relation controls' \
  --output results/isetrace/e2rel-report.md
```

`--name-prefix` discovers matching `runs/<name>/` directories and adds selected-epoch and graph-control diagnostics. Use repeated `--run 'runs/<glob>'` arguments when reporting legacy names that predate this convention.

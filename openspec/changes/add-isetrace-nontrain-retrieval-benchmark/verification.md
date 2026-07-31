# Verification

## Structural pilot preparation

Command:

```bash
uv run python - <<'PY'
from pathlib import Path
from graph_memory.datasets.isetrace import prepare_isetrace_benchmark
b, s = prepare_isetrace_benchmark(
    Path('data/isetrace/query-authoring/test-pilot-100.jsonl'),
    Path('data/isetrace/splits/v1/test.trajectories.jsonl'),
    source_revision='e40e04d41c04e4eb4bae181ebdd41b61c688081b',
    count=None, seed=13, offset=0, strict=True,
    review_policy='allow_unreviewed', label_policy='intent_aware',
)
print(len(b.rankings), len(b.labels), len(b.provenance_graphs))
print(s.to_dict())
PY
```

Observed:

```text
100 100 61
candidate_outputs=2858
unique_candidate_outputs=1754
path_supported_tasks=33
review_status_unreviewed=100
```

All graph/query/output/dependency joins passed strict validation. This is structural evidence only; all 100 records remain unreviewed.

## End-to-end runs

Completed successfully:

```text
BM25 full 100-query pilot:
  runs/isetrace-bm25-pilot/
  100 rankings and execution_provenance_v1 evaluation

Dense smoke:
  runs/isetrace-dense-smoke/

GraphRAG title-free smoke:
  runs/isetrace-graphrag-smoke-v3/

Provenance-path smoke:
  runs/isetrace-provenance-path-smoke-v2/
```

Every run used the standard Hydra -> Prefect -> content-addressed artifacts -> evaluation -> run-output path. Smoke/full metrics are engineering evidence and are not paper results.

The first live title-free GraphRAG artifact reload exposed optional Pydantic fields that lacked `= None` defaults while publication uses `exclude_none=True`. The contracts were corrected and a round-trip regression test was added; the subsequent live GraphRAG run completed.

## Automated checks

```text
uv run pytest -q
  131 passed

Focused post-fix suite:
  23 passed

uv run ruff check .
  All checks passed

uv run basedpyright --level error
  0 errors, 0 warnings, 0 notes

uv run python -m compileall -q graph_memory experiment scripts tests
  passed

git diff --check
  passed
```

Focused behavior covers:

- one graph reused by multiple queries without fingerprint changes;
- review filtering and intent-aware evidence targets;
- feeds and artifact lifecycle dependency projection;
- reverse and multi-hop provenance completion;
- temporal-only exact Dense fallback;
- stored-direction edge emission;
- title-free shared-entity GraphRAG intervention;
- closed native-trace serialization without optional-field loss;
- aligned BM25, Dense, GraphRAG, and provenance-path stage/evaluation outputs.

## Pending scientific gates

- Manual acceptance/edit/rejection of the natural-query pilot.
- Formal `accepted_only` query artifact.
- Full four-method experiment on the accepted artifact.
- Per-intent and complete-chain subset analysis.
- Paper table/text update only after those runs.

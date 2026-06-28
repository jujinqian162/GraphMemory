# Memory Stream Search Space

Date: 2026-06-15

Status: current Memory Stream tuning config contract.

## Purpose

`configs/search_spaces/memory_stream.json` defines the candidate scoring configs for `scripts/tune_memory_stream.py` and the workflow tune stage. The search-space file controls which fields are fixed and which fields are searched.

The tuner evaluates candidates on the dev split and writes:

```text
<output_config>
<output_config stem>.candidates.json
<output_config stem>.run_summary.json
```

The selected config is later consumed by test retrieval through `RetrieveIO.selected_config`.

## Fields

Each field is required and must be a non-empty array:

| Field | Meaning |
|---|---|
| `relevance_weight` | Weight for dense seed relevance scores. |
| `recency_weight` | Weight for request-owned recency scores. LongMemEval uses latest-visible real-time day decay; legacy position requests use pseudo-recency. |
| `importance_weight` | Weight for Memory Stream importance scores. LongMemEval phase 1 fixes this to `0.0` unless a non-gold external artifact is explicitly configured. |
| `recency_decay` | Decay used before recency normalization: per day for `recency_mode=real_time`, per position step for legacy position recency. |

All weights must be finite and non-negative. At least one of `relevance_weight`, `recency_weight`, and `importance_weight` must be positive for every candidate. `recency_decay` must satisfy `0 < recency_decay <= 1.0`.

When `--importance` is omitted, Memory Stream tuning uses the importance maps already present in `TemporalMemoryRankingRequest`. If any candidate config has `importance_weight > 0.0`, every temporal request must cover every candidate with non-gold importance. LongMemEval V1 phase-1 configs therefore keep `importance_weight` fixed to `0.0`.

## Fixed Fields

There is no special code path for fixed fields. Use a single-element array:

```json
{
  "relevance_weight": [1.0],
  "recency_weight": [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
  "importance_weight": [0.0],
  "recency_decay": [0.99]
}
```

This LongMemEval phase-1 shape searches `recency_weight` over real-time temporal requests; the other fields are fixed by configuration.

## Boundary

The generic grid-search layer does not understand these field names. It only expands the arrays into candidate records. `memory_stream_grid_from_record()` parses each expanded record into `MemoryStreamScoringConfig`, which is the same scoring config type used by formal Memory Stream retrieval.

Do not encode assumptions such as `relevance_weight=1.0` in Python code. If a run should fix that value, keep the single-element array in this JSON file.

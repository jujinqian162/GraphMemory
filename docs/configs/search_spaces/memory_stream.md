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
| `recency_weight` | Weight for pseudo-recency scores. |
| `importance_weight` | Weight for cleaned Memory Stream importance scores. |
| `recency_decay` | Decay used to compute pseudo-recency before normalization. |

All weights must be finite and non-negative. At least one of `relevance_weight`, `recency_weight`, and `importance_weight` must be positive for every candidate. `recency_decay` must satisfy `0 < recency_decay <= 1.0`.

## Fixed Fields

There is no special code path for fixed fields. Use a single-element array:

```json
{
  "relevance_weight": [1.0],
  "recency_weight": [0.0],
  "importance_weight": [0.0, 0.01, 0.05, 0.1],
  "recency_decay": [0.99]
}
```

This searches only `importance_weight`; the other fields are fixed by configuration.

## Boundary

The generic grid-search layer does not understand these field names. It only expands the arrays into candidate records. `memory_stream_grid_from_record()` parses each expanded record into `MemoryStreamScoringConfig`, which is the same scoring config type used by formal Memory Stream retrieval.

Do not encode assumptions such as `relevance_weight=1.0` in Python code. If a run should fix that value, keep the single-element array in this JSON file.

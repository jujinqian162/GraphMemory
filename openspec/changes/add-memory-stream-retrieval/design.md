# Design: Memory Stream Retrieval

## Context

The baseline uses dense relevance, pseudo-recency from item position, and
offline importance. Existing importance scores were produced manually in
multiple batches and exhibit scorer-scale drift. Retrieval needs a stable
consumer artifact without obsolete model-generation machinery.

## Data Preparation

`scripts/data/clean_importance.py` reads canonical dev tasks and the legacy
importance artifact. It selects the first 1000 canonical tasks by default and
requires exact task count, order, task ids, content digests, and node coverage.

For each task, sorted unique raw score levels are mapped evenly to integers
`1..10` using half-up rounding. Equal raw scores remain equal and strict order
is preserved. A constant task maps to `5`.

```json
{
  "schema_version": 1,
  "method": "memory_stream",
  "tasks": [
    {
      "task_id": "hotpot_x",
      "content_digest": "<sha256>",
      "scores": {"m0": 10, "m1": 1}
    }
  ]
}
```

The summary records source/output hashes, legacy source metadata, global
distributions, 40-task source-shard distributions, and anomaly lists. It is
provenance only and is not consumed by retrieval.

## Retrieval

For each task:

1. Compute dense relevance for every memory item.
2. Compute `recency_decay ** (max_position - position)`.
3. Read the validated normalized importance score.
4. Min-max normalize each signal independently within the task.
5. Apply non-negative weights and rank by `(-score, node_id)`.

The artifact may contain extra tasks. Subset selection joins by `task_id`,
rejects duplicate or missing records, and validates content digest and exact
node coverage before ranking.

## Ownership

Importance is an external read-only data dependency. The workflow does not
create an annotation stage, invoke a model, own a cache, or copy importance
data into run-local delivery. Retrieval provenance records artifact path/hash.

## Verification

- Unit tests cover schema validation, subset selection, rank normalization,
  ties, constants, idempotence, and mismatch failures.
- Real-data cleaning must produce 1000 tasks and 41185 scores.
- Repository search must find no active annotation runtime, prompt, cache, or
  model configuration.
- OpenSpec strict validation and repository quality gates must pass.

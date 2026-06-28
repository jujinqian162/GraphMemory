# Design: Memory Stream Retrieval

## Context

The baseline uses dense relevance, request-owned recency, and
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

## Split Policy

The experiment manifest continues to define a shared default split policy for
the ordinary methods. That shared policy remains strict: if a split requests
more data than exists, the workflow should still fail rather than silently
truncate it.

Memory Stream gets two explicit exceptions. First, an experiment may set
`split_sources.dev` or `split_sources.test` to `"importance"`. That source does
not construct tasks from the compact importance artifact alone. Instead, the
prepare stage reads the cleaned importance task ids in artifact order, joins
them against canonical processed HotpotQA input and label artifacts, verifies
the joined input content digest against the importance record, and writes
run-local split files. The compact importance artifact is therefore a split
selector, while the canonical processed input/label files remain the source of
complete `MemoryTaskInput` and label records.

Second, when the selected profile asks for a larger test set than the cleaned
importance sidecar can cover, the workflow caps Memory Stream to the available
covered prefix, emits a warning, and writes the capped count into the
method-specific stage config and run summary. The cap is method-scoped only; it
does not change the default split for other methods.

## Retrieval

The retrieval stage owns file IO. `RetrieveIO.importance` points to the cleaned
artifact. `scripts/run_retrieval.py` reads it once before method construction,
and the registry builder calls `select_importance_records()` against the
current retrieval task list. This accepts an artifact superset but rejects
missing tasks, duplicate task ids, stale content digests, and node mismatch.

The builder constructs one `task_id -> TaskImportanceRecord` index and injects
it together with an existing dense seed ranker into `MemoryStreamMethod`.
The method performs no path or JSON IO. For each task it:

1. Calls the injected dense seed ranker for raw relevance over all nodes.
2. Computes request-owned recency: `recency_decay ** age_days` from the latest visible temporal anchor for `recency_mode=real_time`, or `recency_decay ** (max_position - position)` for legacy position requests.
3. Looks up the validated cleaned importance score by task and node id.
4. Min-max normalizes each signal independently within the task.
5. Applies non-negative weights and sorts by `(-score, node_id)`.

The artifact may contain extra tasks. Subset selection joins by `task_id`,
rejects duplicate or missing records, and validates content digest and exact
node coverage before ranking.

Use a Memory Stream-owned normalization/scoring module rather than importing
the graph-rerank package's private normalization implementation. Constant
signals map to `0.0`. Settings require non-negative weights, at least one
positive weight, and `0 < recency_decay <= 1`.

## Registry and Stage Integration

Add `RetrievalMethodId.MEMORY_STREAM` and
`MemoryStreamRetrievalSettings(top_k, encoder, relevance_weight,
recency_weight, importance_weight, recency_decay)`. Add a dedicated
`MemoryStreamBuildPayload(task_inputs, importance_artifact, importance_path,
importance_sha256, dense_encoder)` containing the loaded compact artifact and
an optional injected dense encoder.

Implement `MemoryStreamMethod` in
`graph_memory/retrieval/methods/memory_stream/method.py` and its pure min-max
and weighted-score helpers in
`graph_memory/retrieval/methods/memory_stream/scoring.py`. The settings
dataclass validates weights and decay in `__post_init__`.

Memory Stream uses `RetrievalLifecycle.STATELESS`, with its encoder sourced
from experiment config. It reuses `STATELESS_RETRIEVAL_WORKFLOW`; no workflow
id, stage id, artifact role, or run-local importance artifact is added.

`RetrieveIO` gains optional `importance: Path | None`. Stage-config compilation
sets it only for Memory Stream, using the
`memory_stream_importance_path` experiment-config override when supplied and
otherwise
`data/hotpotqa/processed/memory_stream/dev.first_1000.importance.json`.
`run_retrieval.py` fails with the concrete path if it is absent, records it in
the summary inputs, and computes its SHA-256 before method construction.

The stage-config path that owns Memory Stream also owns the method-specific
test cap. It must derive the truncated test count before `RetrieveIO` and
`EvaluateIO` are written so both stages consume the same capped inputs.

Add `ImportanceArtifactProvenance(path: Path, sha256: str,
schema_version: int)` and optional
`RetrievalProvenance.importance: ImportanceArtifactProvenance | None`.
Artifact loading, subset validation, and hashing happen before
`run_retrieval()` starts its per-task latency timer.

## Ownership

Importance is an external read-only data dependency. The workflow does not
create an annotation stage, invoke a model, own a cache, or copy importance
data into run-local delivery. Retrieval provenance records artifact path/hash.

## Verification

- Unit tests cover schema validation, subset selection, rank normalization,
  ties, constants, idempotence, and mismatch failures.
- Method tests use a fake dense seed ranker and prove complete-node output,
  exact score combination, constant-signal handling, and node-id tie-breaking.
- Registry/stage tests prove one-time artifact loading, pre-timing validation,
  stateless workflow reuse, default/override paths, and serialized provenance.
- Real-data cleaning must produce 1000 tasks and 41185 scores.
- Repository search must find no active annotation runtime, prompt, cache, or
  model configuration.
- OpenSpec strict validation and repository quality gates must pass.

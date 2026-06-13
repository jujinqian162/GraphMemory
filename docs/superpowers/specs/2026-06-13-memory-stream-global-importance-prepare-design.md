# Memory Stream Global Importance Prepare Design

Date: 2026-06-13

Status: Approved

## Problem

Memory Stream importance annotation was incorrectly modeled as a workflow stage
that produced run-local files below `runs/<experiment_name>/`. Importance is
query-independent, expensive, and reusable across experiment profiles and
workspaces. Its producer must therefore be a one-time global preprocessing
command, analogous to dataset download and canonical dataset preparation.

## Ownership

`scripts/annotate_importance.py` is a standalone preprocessing CLI. It is not
registered as a workflow stage, is not planned by `scripts/experiment.py`, and
does not receive run-local paths from a workflow manifest.

The default command is:

```powershell
python scripts/annotate_importance.py
```

It uses these defaults from the repository root:

```text
tasks:       data/hotpotqa/processed/dev_memory_tasks.input.json
output:      data/hotpotqa/processed/memory_stream/dev.importance.json
summary:     data/hotpotqa/processed/memory_stream/dev.importance.run_summary.json
cache:       data/cache/memory_stream_importance/
model id:    Qwen/Qwen2.5-7B-Instruct
model path:  models/Qwen2.5-7B-Instruct
device:      auto
max tokens:  2048
```

Every default remains overrideable through CLI arguments. When `--output` is
overridden without `--summary`, the summary is derived beside the selected
output.

## Data Flow

```text
data/hotpotqa/processed/dev_memory_tasks.input.json
  -> scripts/annotate_importance.py
  -> graph_memory.retrieval.methods.memory_stream.annotation
  -> data/cache/memory_stream_importance/<prefix>/<digest>.json
  -> data/hotpotqa/processed/memory_stream/dev.importance.json
  -> data/hotpotqa/processed/memory_stream/dev.importance.run_summary.json
```

The CLI validates the complete canonical task input, scans the cache before
loading the model, loads one local Transformers runtime only when cache misses
exist, processes misses sequentially, and atomically replaces the final
artifact only after all tasks succeed.

## Shared Consumption Contract

The global artifact contains the complete canonical dev task set in canonical
order. A workflow may consume any subset of those tasks.

Subset consumption joins by `task_id` and validates the selected task's
`content_digest` and exact node-id coverage. Extra tasks in the global artifact
are allowed. Missing task ids, duplicate artifact task ids, changed content, or
changed node coverage fail before retrieval.

The producer still validates exact full-input order and coverage before writing
the final artifact.

## Workflow Boundary

The workflow does not:

- define an `importance` stage;
- invoke `scripts/annotate_importance.py`;
- allocate a run-local importance artifact;
- compile an annotation stage config;
- own annotation status, resume, or cache pruning;
- copy the global cache into a run delivery.

Later Memory Stream retrieval work will treat the global importance artifact as
a read-only external dependency and record its path and semantic metadata in
retrieval provenance.

## Failure Behavior

- A missing task input or model path fails with the concrete path.
- Invalid task input fails before model construction.
- Invalid generated JSON fails the current run without replacing a previously
  successful final artifact.
- Successful per-task cache entries remain reusable after a later failure.
- All-cache-hit reruns construct no Transformers runtime.

## Verification

Automated tests cover zero-argument defaults, CLI overrides, one-process model
lifecycle, cache-only reruns, failed-run preservation, full artifact validation,
subset selection, and the absence of annotation from workflow enums, manifests,
plans, and experiment configuration.


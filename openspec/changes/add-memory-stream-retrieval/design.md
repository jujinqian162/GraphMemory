## Context

The original experiment plan requires a simplified Memory Stream baseline with
`relevance + recency + importance`. HotpotQA has no true event timestamps or
memory-access history, so recency must be a position-derived pseudo-recency.

The available MetaX environment has a verified direct Transformers path.
Qwen2.5-7B-Instruct startup takes roughly 80 seconds, making per-task or
per-workflow loading unacceptable. Importance is query-independent and can be
shared by every smoke, quick, and full workflow derived from the same canonical
HotpotQA dev corpus.

## Goals / Non-Goals

**Goals:**

- Generate a complete global 1-10 importance artifact once.
- Make the default operator command `python scripts/annotate_importance.py`.
- Load the local tokenizer/model at most once per annotation process.
- Preserve successful per-task cache entries across failures and reruns.
- Exclude query, answer, labels, gold nodes, and graph data from prompts and
  semantic cache keys.
- Let later workflow subsets consume selected records from the global artifact.
- Keep timed retrieval independent from the local causal LLM runtime.

**Non-Goals:**

- Adding an importance workflow stage or run-local importance artifact.
- Reproducing the complete Generative Agents simulator.
- Treating `position` as a real timestamp.
- Fine-tuning Qwen or using gold evidence to calibrate importance.
- HTTP/cloud APIs, vLLM, tensor parallelism, multiprocessing, or multi-GPU
  sharding.
- Generating train-split importance for this non-trained baseline.

## Decisions

### Importance remains a sidecar artifact

The global artifact contains model identity, prompt version, semantic generation
settings, and per-task node-score mappings. `MemoryItem` is not modified.

```json
{
  "method": "memory_stream",
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "prompt_version": "memory-stream-importance-v2",
  "generation": {
    "do_sample": false,
    "use_cache": true,
    "max_new_tokens": 2048
  },
  "tasks": [
    {
      "task_id": "example-id",
      "content_digest": "<sha256>",
      "scores": {
        "m0": 7
      }
    }
  ]
}
```

### Annotation is global preprocessing, not workflow execution

The command runs independently from `scripts/experiment.py`:

```text
python scripts/annotate_importance.py
```

Default paths are:

```text
tasks       data/hotpotqa/processed/dev_memory_tasks.input.json
output      data/hotpotqa/processed/memory_stream/dev.importance.json
summary     data/hotpotqa/processed/memory_stream/dev.importance.run_summary.json
cache       data/cache/memory_stream_importance/
model path  models/Qwen2.5-7B-Instruct
```

The CLI owns these IO arguments and permits explicit overrides. There is no
`Registry.configs.IMPORTANCE`, `StageId.IMPORTANCE`, workflow command builder,
manifest artifact allocation, or run-local annotation config.

### One task is one semantic annotation unit

Each generation call contains ordered `{node_id, source, text, position}`
records for one task. The response contains an ordered score array with exactly
one integer in `[1, 10]` for each input item. The parser maps each score back to
the corresponding node id. Wrong-length arrays, booleans, floats, strings, and
out-of-range values fail the run. This avoids requiring the model to reproduce
dozens of node-id strings exactly.

### Cache and runtime lifecycle

The semantic cache key includes model id, prompt version, generation settings,
and ordered item id/source/text/position. It excludes query, labels, model path,
physical device, and run directory.

The process scans all cache entries before model construction. All-cache-hit
runs load no model. Any misses create exactly one local Transformers runtime,
load once, and process every miss sequentially through the same instance.

Before model-facing imports, the CLI clears distributed rank/master variables
and sets `ACCELERATE_USE_DEEPSPEED=false`. CUDA uses local device zero after the
operator selects one physical device with `CUDA_VISIBLE_DEVICES`.

### Global artifact supports workflow subsets

The producer validates exact order and full coverage against the canonical dev
input. A consumer may request a subset in any order. Consumer validation:

1. validates artifact metadata;
2. rejects duplicate artifact task ids;
3. joins requested tasks by `task_id`;
4. rejects missing tasks;
5. validates each requested content digest and exact node-score coverage;
6. returns records in requested workflow task order.

Extra canonical tasks in the global artifact are valid.

### Retrieval remains a later milestone

Later `MemoryStreamMethod` work will reuse the existing dense relevance path,
derive pseudo-recency from item position, normalize all three signals within a
task, and consume the global importance sidecar without importing Transformers.

The future workflow remains:

```text
prepare -> graphs -> retrieve -> evaluate -> aggregate
```

The retrieve command receives the external global artifact path. No annotation
command appears in the plan.

## Risks / Trade-offs

- Qwen scores may vary across runtime versions. Record model id, prompt version,
  generation settings, content digests, and the final artifact.
- A malformed response can stop a long run. Persist each successful task cache
  atomically and write the final artifact only after complete success.
- Canonical dev input changes invalidate selected records through content
  digests.
- A workflow may select tasks in a different order from the canonical artifact.
  Join by task id and revalidate content instead of relying on global order.

## Migration Plan

1. Keep the implemented prompt, cache, runtime, annotation, and atomic IO core.
2. Move annotation settings out of the workflow stage-config registry.
3. Replace the config-only annotation adapter with the standalone defaulted CLI.
4. Remove importance workflow stage, run-local artifacts, planner integration,
   experiment annotation config, and premature Memory Stream method registration.
5. Add subset-safe global artifact selection.
6. Implement retrieval and external-dependency workflow consumption later.

## Open Questions

None. The prepare milestone uses the canonical dev input and global data paths
shown above.

## Context

The original experiment plan requires a simplified Generative Agents Memory Stream baseline with `relevance + recency + importance`. The current repository already provides the shared task/ranking contracts, a batched SentenceTransformer relevance path, typed retrieval builders, strict stage-root configs, workflow manifests, artifact status/resume, and leakage-safe separation between task inputs and labels.

HotpotQA does not provide event timestamps or memory-access history, so the baseline cannot reproduce dynamic Generative Agents recency. The available deployment target is eight MetaX C500 accelerators with 64 GiB each and MACA 3.3.0.x. On this machine, the verified inference path is direct `transformers` loading through `AutoTokenizer` and `AutoModelForCausalLM`; model startup takes about 80 seconds and must not occur per task.

The expensive operation is query-independent importance generation. It must run once, be cached, and remain outside timed retrieval. Relevance remains query-dependent and is computed during retrieval with the existing dense encoder.

## Goals / Non-Goals

**Goals:**

- Add a reproducible `memory_stream` baseline matching the documented three-signal formula.
- Generate 1-10 importance scores with a local Qwen2.5-7B-Instruct service without exposing queries or labels.
- Make annotation restartable and reusable across experiment runs.
- Keep importance generation and retrieval as separate workflow stages and artifacts.
- Reuse existing dense batching, ranking output, evaluation, aggregation, observability, and experiment commands.
- Load the local tokenizer/model at most once per annotation process and keep it resident while all cache misses are processed.
- Preserve the verified MetaX-safe loading settings: single visible GPU, distributed environment disabled, `torch_dtype="auto"`, explicit `device_map`, `low_cpu_mem_usage=True`, and `tp_plan=None`.
- Make all output deterministic after the importance artifact has been created.

**Non-Goals:**

- Reproducing the complete Generative Agents simulator, reflection, planning, memory creation, or dynamic access updates.
- Treating HotpotQA `position` as a real timestamp.
- Fine-tuning Qwen, training an importance model, or using gold evidence to calibrate importance.
- Calling HTTP or cloud LLM APIs.
- Running vLLM, an OpenAI-compatible server, tensor parallelism, multiprocessing, or multi-GPU sharding in the first implementation.
- Installing or replacing the server's proven vendor-compatible `torch`/`transformers` environment as part of the annotation command.
- Generating train-split importance for a non-trained baseline.
- Adding Memory Stream weight tuning or an ablation suite in the first implementation.

## Decisions

### Importance is a sidecar artifact, not a `MemoryItem` field

The annotation stage writes one final JSON artifact containing semantic model identity, prompt version, generation settings, and per-task node-score mappings. Runtime configuration keeps `model_id` separate from `model_path`: the id enters artifacts and cache keys, while the path is only used to load local files. `MemoryItem` remains the dataset-visible source contract and is not rewritten.

The artifact has this logical shape:

```json
{
  "method": "memory_stream",
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "prompt_version": "memory-stream-importance-v1",
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
        "node-id": 7
      }
    }
  ]
}
```

The validator requires exact task alignment, exact node-id coverage, matching ordered memory-item content digests, no extra nodes, and integer scores in `[1, 10]`.

Alternative considered: add `importance` to every `MemoryItem`. Rejected because that would mix generated method-specific state into the common dataset contract and force all methods to carry it.

### Annotation is a first-class workflow stage

Add `StageId.IMPORTANCE` to both the stage-config registry enum and workflow enum, place it between graph construction and retrieval in workflow ordering, and add `ArtifactRole.IMPORTANCE_SCORES`. The Memory Stream workflow is:

```text
prepare -> graphs -> importance -> retrieve -> evaluate -> aggregate
```

Graph construction remains present because the shared evaluation workflow consumes graph artifacts for connectivity metrics; Memory Stream retrieval itself does not consume graphs.

The low-level command is always:

```text
scripts/annotate_importance.py --config <stage-config>
```

The stage receives only the test `*_memory_tasks.input.json`, never labels. The retrieve stage receives the resulting sidecar path. Annotation time is recorded in its own run summary and is not added to `RankedResult.latency_ms`.

Alternative considered: lazily call the LLM inside `run_retrieval.py`. Rejected because it destroys latency comparability, makes retries expensive, and prevents clean cache/status semantics.

### Memory Stream configuration is compiled into stage-root configs

The experiment config gains one fixed `memory_stream` section containing annotation, cache, relevance encoder, score weights, and recency settings. Workflow initialization validates this section when `memory_stream` is selected and compiles:

- `ImportanceStageConfig` for the annotation script.
- `RetrieveStageConfig` with `MemoryStreamRetrievalSettings` and an importance artifact input.

Low-level scripts do not reload experiment config. The existing trainable-only `Registry.configs.TRAINABLE_METHOD` contract remains unchanged.

Alternative considered: broaden `TrainableMethodConfig` into a union containing Memory Stream. Rejected because Memory Stream has no pair/train lifecycle and would weaken the recently established trainable configuration boundary.

### The annotation stage owns one persistent local Transformers runtime

Before importing repository modules that may transitively import Torch, `scripts/annotate_importance.py` removes `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`, and `MASTER_PORT` from the process environment and sets `ACCELERATE_USE_DEEPSPEED=false`. The operator selects one physical GPU before process startup with `CUDA_VISIBLE_DEVICES`; inside the process the selected accelerator is local device 0.

The annotation service scans and validates all task cache entries before constructing the model runtime. If every task is a cache hit, the stage completes without importing `transformers` or loading model weights. If one or more tasks miss, it creates exactly one `LocalTransformersImportanceRuntime`, loads tokenizer and model once from `model_path`, processes every miss through that same instance, and releases it only when the stage exits.

The runtime follows the verified loading path:

```python
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
device_map = {"": 0} if torch.cuda.is_available() else {"": "cpu"}
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map=device_map,
    low_cpu_mem_usage=True,
    tp_plan=None,
)
model.eval()
```

Generation uses `tokenizer.apply_chat_template()` when available, moves tokenized inputs to the model parameter device, executes under `torch.inference_mode()`, and calls `model.generate()` with `do_sample=False`, `use_cache=True`, configured `max_new_tokens`, and a valid pad token. CUDA synchronization brackets timing measurements.

Alternative considered: start a local HTTP/vLLM server or load a model for each task. Rejected because HTTP serving is not known to work in this environment and repeated model startup would add roughly 80 seconds per task.

### One task is one semantic annotation unit

Each generation call contains an ordered list of `{node_id, source, text, position}` records for one task and asks for absolute 1-10 importance scores. The prompt never contains `query`, labels, answers, gold nodes, or graph edges. The response must be one JSON object with exactly one integer score per node id.

The stage strips an optional Markdown code fence and parses JSON. Missing keys, extra keys, duplicate node ids, non-integers, booleans, and out-of-range scores are invalid. A malformed deterministic generation fails with the task id and leaves earlier successful cache entries available for restart; the stage does not reload the model or apply a silent score-4 fallback.

Alternative considered: rate every sentence in a separate generation call. Rejected because it multiplies generation count by roughly forty while providing no required benchmark benefit.

### Cache keys are content-addressed and query-independent

For each task, canonical JSON is built from:

- model id;
- prompt version;
- generation parameters that affect output;
- the ordered memory item fields `id`, `source`, `text`, and `position`.

The query and labels are excluded. SHA-256 of this canonical payload is the cache key. Each validated task result is written atomically to `<cache_dir>/<prefix>/<digest>.json`. The final artifact is assembled in task input order from cache hits and new results.

Changing `model_id`, prompt version, generation settings, item content, item order, or node ids creates a new key. Changing `model_path`, physical GPU selection, or other non-semantic runtime placement does not invalidate semantic cache entries. Operators must change `model_id` when replacing the weights behind a local path.

Alternative considered: cache only by node text. Rejected because one-task prompts introduce task context and node-id coverage must remain exact.

### Scoring uses deterministic three-signal normalization

`MemoryStreamMethod` receives:

- the existing `DenseTaskRetriever` as the relevance ranker;
- validated importance scores for all selected tasks;
- `relevance_weight`, `recency_weight`, `importance_weight`;
- `recency_decay` in `(0, 1]`.

For each memory item:

```text
relevance_raw = dense cosine score
age_steps = max(position) - position
recency_raw = recency_decay ** age_steps
importance_raw = integer importance in [1, 10]
```

Each signal is min-max normalized across all items in the task. A constant signal maps to all zeros because it carries no ranking information. The final score is:

```text
score =
    relevance_weight * relevance_normalized
  + recency_weight * recency_normalized
  + importance_weight * importance_normalized
```

Defaults are equal weights and `recency_decay = 0.99`. Results are sorted by descending score and then ascending `node_id`. Retrieved edges remain empty.

Alternative considered: use raw `(importance - 1) / 9` while min-max normalizing other signals. Rejected because the original Memory Stream retrieval normalizes all components before combination and the baseline should preserve that behavior.

### Method metadata declares the importance dependency

Add `RetrievalLifecycle.MEMORY_STREAM` and an `ImportanceSource` field to `RetrievalDependencySpec`. `memory_stream` declares:

- no graph input for retrieval;
- no tuned graph config;
- no checkpoint/model artifact;
- experiment-config dense encoder;
- sidecar importance artifact.

The workflow registry maps this lifecycle to `MEMORY_STREAM_WORKFLOW`. The retrieval builder receives a dedicated `MemoryStreamBuildPayload`, validates task/artifact alignment, constructs the dense relevance ranker, and returns provenance containing the encoder and importance artifact metadata.

Alternative considered: classify Memory Stream as `STATELESS`. Rejected because it has a required preprocessing artifact and distinct workflow/status dependencies.

### Tests never require a live accelerator or real LLM

Unit and stage tests inject a fake persistent runtime and fake runtime loader plus a fake sentence encoder. They prove that all-cache-hit runs load zero models, mixed runs load exactly one model, every cache miss reuses the same runtime instance, malformed output does not trigger model reload, and stage shutdown happens once. They also cover prompt leakage, exact response validation, cache/resume, normalization, ties, artifact alignment, builder dispatch, stage configs, workflow planning, status, and run summaries.

A documented operator smoke test uses the proven direct Transformers script path on one MetaX C500, but it is not part of the default pytest suite.

## Risks / Trade-offs

- [Qwen scores can vary across runtime versions even with deterministic generation settings] -> Record model identity, prompt version, generation settings, content digests, and cache results; treat the sidecar as the reproducible experimental input.
- [Batching all task items can create relative-context bias] -> Prompt for absolute independent ratings and keep one stable prompt version; do not claim exact equivalence to one-event Generative Agents annotation.
- [Pseudo-recency is not temporal recency] -> Name and document it explicitly, derive it only from `position`, and avoid claims about real memory access.
- [Model startup costs about 80 seconds] -> Scan cache first, load only when misses exist, instantiate exactly once, and report `model_load_seconds` separately.
- [A malformed response can stop a long run] -> Persist every successful task atomically and make restart reuse cache entries without regenerating completed tasks.
- [The process can fail after model load] -> Final artifact is written only after every task is validated; partial progress lives only in the content-addressed cache, so restart pays one reload but not completed generations.
- [Distributed environment variables can trigger unsupported launch behavior] -> Remove rank/master variables before Torch/Transformers import, set `ACCELERATE_USE_DEEPSPEED=false`, select one GPU with `CUDA_VISIBLE_DEVICES`, and pass `tp_plan=None`.
- [Concurrent `generate()` calls may be unstable on the vendor stack] -> Process cache misses sequentially through one model instance in the first implementation.
- [Adding a stage affects planner range and status logic] -> Update closed enums, manifest contracts, dependency validation, resume/status tests, and all stage-order assertions together.
- [Importance artifact could accidentally leak query or labels] -> Build canonical prompt payload directly from `memory_items`; add tests that sentinel query/answer/gold values never appear in prompts or cache keys.

## Migration Plan

1. Add contracts, validation, prompt construction, fake-runtime tests, and content-addressed cache support.
2. Add the annotation stage config, stage runner, CLI, run summary, and restart behavior.
3. Add Memory Stream normalization, method implementation, retrieval settings, builder payload, and provenance.
4. Register method metadata and compile importance/retrieve stage configs.
5. Add workflow artifacts, planner commands, dependency checks, status/resume, manifest validation, and delivery collection.
6. Add the experiment configuration section and operations documentation for model download, direct Transformers loading, environment cleanup, one-GPU execution, cache-aware annotation, and retrieval.
7. Run focused tests, full tests, lint/type checks, experiment planning smoke, and strict OpenSpec validation.

Rollback is a normal source-control revert plus deletion of Memory Stream run directories. The reusable cache is method-specific and can be removed independently; existing methods and artifacts require no migration.

## Open Questions

None. The first implementation uses fixed equal weights, one persistent single-GPU Qwen2.5-7B-Instruct runtime, one task per deterministic generation call, and position-derived pseudo-recency.

## ADDED Requirements

### Requirement: Importance annotation is query-independent and leakage-safe
The system SHALL construct Memory Stream importance prompts only from ordered memory item ids, sources, texts, and positions.

#### Scenario: Query and labels are excluded
- **WHEN** a task contains sentinel values in its query, answer, gold evidence nodes, or graph data
- **THEN** none of those values appear in the importance prompt payload, semantic cache key, or importance artifact

#### Scenario: Source and text are available to the annotator
- **WHEN** an importance prompt is constructed for a task
- **THEN** each memory item is represented by its node id, source, text, and position

### Requirement: Local LLM responses provide exact ordered integer importance coverage
The system SHALL accept an annotation response only when it contains an ordered array with exactly one integer score in the inclusive range 1-10 for every input memory item.

#### Scenario: Valid response is accepted
- **WHEN** the local LLM returns a JSON score array whose length matches the ordered input items and whose values are integers from 1 through 10
- **THEN** each score is mapped to the corresponding input node id and the task annotation may be cached

#### Scenario: Missing or extra scores are rejected
- **WHEN** a response contains fewer or more scores than input memory items
- **THEN** validation fails with the task id and expected and observed counts

#### Scenario: Invalid score types or ranges are rejected
- **WHEN** a response contains a boolean, non-integer, value below 1, or value above 10
- **THEN** validation fails with the task id and offending node id

### Requirement: Global annotation owns one persistent local model lifecycle
The system SHALL load the tokenizer and causal language model at most once in the standalone annotation process and SHALL reuse the same model instance for every cache miss.

#### Scenario: All cache hits avoid model loading
- **WHEN** every task has a valid matching cache entry
- **THEN** annotation completes without importing or loading the Transformers model runtime

#### Scenario: Cache misses trigger one model load
- **WHEN** one or more tasks are missing valid cache entries
- **THEN** the tokenizer and model are loaded exactly once before the first generation and remain resident until all misses have been processed

#### Scenario: Every miss uses the same runtime
- **WHEN** multiple tasks require generation
- **THEN** each task is processed sequentially through the same tokenizer and model objects without per-task load or unload

#### Scenario: Invalid output does not reload the model
- **WHEN** deterministic generation for a task produces an invalid response
- **THEN** the stage fails with the task id, keeps earlier cache entries, and does not construct a second model instance

### Requirement: Local generation follows the proven single-device Transformers path
The system SHALL prepare a single-process environment and use direct `AutoTokenizer` and `AutoModelForCausalLM` generation with tensor parallelism disabled.

#### Scenario: Distributed launch state is cleared before model import
- **WHEN** annotation starts with rank or master environment variables present
- **THEN** it removes `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`, and `MASTER_PORT` and sets `ACCELERATE_USE_DEEPSPEED=false` before Torch or Transformers model loading

#### Scenario: Visible GPU zero is used inside the process
- **WHEN** CUDA is available and the operator selected one physical device through `CUDA_VISIBLE_DEVICES`
- **THEN** model loading uses local `device_map={"": 0}` and explicitly passes `tp_plan=None`

#### Scenario: Generation is deterministic
- **WHEN** an importance prompt is generated
- **THEN** the runtime uses chat-template formatting when available, `torch.inference_mode()`, `do_sample=False`, `use_cache=True`, configured `max_new_tokens`, and a valid pad token id

### Requirement: Per-task importance results are content-addressed and restartable
The system SHALL cache each validated task annotation under a SHA-256 key derived from model id, prompt version, semantic generation settings, and ordered memory item content while excluding query and labels.

#### Scenario: Matching cache entry avoids model generation
- **WHEN** a valid cache entry matches the current semantic cache key and exact node coverage
- **THEN** the stage reuses its scores without a model generation call

#### Scenario: Semantic input change invalidates the cache
- **WHEN** model id, prompt version, relevant generation settings, item id, source, text, position, or item order changes
- **THEN** the stage computes a different cache key and performs a new local generation

#### Scenario: Runtime placement change does not invalidate scores
- **WHEN** only local model path, physical GPU selection, or other non-semantic device placement changes while model id remains the same
- **THEN** the semantic cache key remains unchanged

#### Scenario: Successful tasks survive a later failure
- **WHEN** earlier tasks have been cached and a later task fails generation or response validation
- **THEN** a subsequent run reuses the earlier valid cache entries

### Requirement: Final global importance artifacts are complete and auditable
The system SHALL write the final global importance artifact only after all canonical tasks have validated scores and SHALL write a run summary describing annotation inputs, outputs, settings, counts, timings, cache behavior, and failures.

#### Scenario: Complete artifact aligns with task input
- **WHEN** annotation succeeds for every task
- **THEN** the final artifact contains model id, prompt version, generation settings, content digest, and exact node scores for every input task

#### Scenario: Partial run does not masquerade as complete
- **WHEN** annotation terminates before all tasks are valid
- **THEN** the final artifact is absent or remains stale and the failed run summary records the error

#### Scenario: Run summary distinguishes cache and model work
- **WHEN** annotation completes with both cache hits and new generations
- **THEN** the run summary reports task count, memory-item count, cache-hit count, model-load count, model-load seconds, generation-call count, generated-token count, generation seconds, and total annotation time

### Requirement: Standalone annotation has executable defaults
The system SHALL allow the global annotation command to run without a config file or required CLI arguments from the repository root.

#### Scenario: Zero-argument command uses canonical paths
- **WHEN** the operator runs `python scripts/annotate_importance.py`
- **THEN** the command reads `data/hotpotqa/processed/dev_memory_tasks.input.json`
- **AND** writes `data/hotpotqa/processed/memory_stream/dev.importance.json`
- **AND** writes its run summary beside the output
- **AND** uses `data/cache/memory_stream_importance` and `models/Qwen2.5-7B-Instruct`

#### Scenario: Output override derives summary
- **WHEN** the operator supplies `--output` without `--summary`
- **THEN** the run summary path is derived beside the selected output

### Requirement: Global artifacts support validated subset selection
The system SHALL allow later workflow task subsets to select records from the complete global artifact by task id.

#### Scenario: Extra canonical tasks are allowed
- **WHEN** the artifact contains more tasks than the requested workflow subset
- **THEN** selected records are returned in workflow task order

#### Scenario: Missing, duplicate, or stale task records fail
- **WHEN** a requested task is missing, an artifact task id is duplicated, or selected content/node coverage differs
- **THEN** validation fails before retrieval

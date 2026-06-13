## 1. Importance Contracts and Validation

- [x] 1.1 Create `graph_memory/retrieval/methods/memory_stream/` and define typed records for generation settings, task importance records, final importance artifacts, annotation results, and cache statistics.
- [x] 1.2 Add failing tests in `tests/test_memory_stream_importance.py` for exact task/node coverage, integer 1-10 validation, duplicate node rejection, content digest validation, and deterministic artifact order.
- [x] 1.3 Implement canonical memory-item serialization and SHA-256 content/cache digests that exclude query, labels, runtime device placement, and graph data.
- [x] 1.4 Add `graph_memory/validation/importance.py`, export its validators through `graph_memory/validation/__init__.py`, and make validation errors identify task and node ids.
- [x] 1.5 Run `uv run pytest tests/test_memory_stream_importance.py -q`.

## 2. Leakage-Safe Prompt and Response Parsing

- [x] 2.1 Add tests proving sentinel query, answer, gold-node, and graph values never enter prompt text or semantic cache keys.
- [x] 2.2 Implement the versioned `memory-stream-importance-v1` system/user prompt from only ordered `{node_id, source, text, position}` records.
- [x] 2.3 Add strict response parsing for plain JSON and one optional Markdown JSON fence.
- [x] 2.4 Reject missing/extra keys, duplicate ids, booleans, floats, strings, and out-of-range values without applying a fallback score.
- [x] 2.5 Run `uv run pytest tests/test_memory_stream_importance.py -q`.

## 3. Persistent Local Transformers Runtime

- [x] 3.1 Add fake-loader tests proving all-cache-hit runs create zero runtimes, any number of cache misses creates exactly one runtime, and every miss reuses the same tokenizer/model instance.
- [x] 3.2 Implement `graph_memory/retrieval/methods/memory_stream/runtime.py` with lazy imports of `torch`, `AutoTokenizer`, and `AutoModelForCausalLM`.
- [x] 3.3 Load with `trust_remote_code=True`, `torch_dtype="auto"`, CUDA `device_map={"": 0}` or CPU fallback, `low_cpu_mem_usage=True`, and explicit `tp_plan=None`, then call `model.eval()`.
- [x] 3.4 Implement chat-template prompt formatting, input movement to the model parameter device, `torch.inference_mode()`, `do_sample=False`, `use_cache=True`, configured `max_new_tokens`, pad-token fallback, CUDA synchronization, generated-token counting, and timing.
- [x] 3.5 Add tests proving invalid generated JSON fails the current task without constructing or loading a second model instance.
- [x] 3.6 Add an architecture test proving annotation code contains no HTTP, OpenAI SDK, vLLM server, thread-pool generation, tensor-parallel, or per-task `from_pretrained()` path.
- [x] 3.7 Run `uv run pytest tests/test_memory_stream_importance.py -q`.

## 4. Content-Addressed Cache and Atomic Writes

- [x] 4.1 Add `write_json_atomic()` to `graph_memory/infrastructure/io.py`, re-export it through `graph_memory/io.py`, and test replacement without exposing partial JSON.
- [x] 4.2 Implement per-task cache paths as `<cache_dir>/<first-two-digest-chars>/<digest>.json`.
- [x] 4.3 Validate every cache hit against model, prompt version, generation settings, content digest, and exact node coverage before reuse.
- [x] 4.4 Add tests for cache hit, semantic invalidation, device-placement-only changes, corrupted cache recovery, and reuse of successful tasks after a later failure.
- [x] 4.5 Run `uv run pytest tests/test_memory_stream_importance.py tests/test_phase1_real_io_observability.py -q`.

## 5. Importance Stage and CLI

- [x] 5.1 Add `StageId.IMPORTANCE` to `graph_memory/registry/ids.py`, add typed `ImportanceIO`, `ImportanceAnnotationSettings`, and `ImportanceStageConfig` to `graph_memory/registry/stage_configs.py`, and register `Registry.configs.IMPORTANCE`.
- [x] 5.2 Add failing stage tests in `tests/test_memory_stream_annotation_stage.py` for config loading, fake-runtime execution, final artifact creation, failed-run behavior, and run-summary fields.
- [x] 5.3 Implement `graph_memory/stages/importance.py` to validate tasks, resolve all cache hits before model construction, load one runtime only when misses exist, process misses sequentially, assemble the final artifact, and return lifecycle counters/timings.
- [x] 5.4 Create `scripts/annotate_importance.py`; before importing Torch/Transformers-facing repository modules, remove `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`, and `MASTER_PORT`, set `ACCELERATE_USE_DEEPSPEED=false`, then load `Registry.configs.IMPORTANCE` and call the stage runner.
- [x] 5.5 Write success/failure run summaries with task/item counts, cache hits, model-load count, model-load seconds, generation calls, generated tokens, generation seconds, model id/path, device, prompt version, generation settings, output path, and total time.
- [x] 5.6 Ensure the final artifact is written only after every task succeeds while successful per-task cache files remain after failure.
- [x] 5.7 Run `uv run pytest tests/test_memory_stream_annotation_stage.py tests/test_registry_stage_configs.py tests/test_cli_contracts.py -q`.

## 6. Three-Signal Memory Stream Ranking

- [ ] 6.1 Add failing tests in `tests/test_memory_stream_retrieval.py` for raw recency, min-max normalization, constant signals, weighted sums, node-id ties, and empty retrieved edges.
- [ ] 6.2 Implement reusable min-max normalization with the constant-input-to-zero rule.
- [ ] 6.3 Implement `MemoryStreamMethod` using an injected `SeedRanker`, validated task importance records, equal default weights, and default `recency_decay=0.99`.
- [ ] 6.4 Verify Memory Stream relevance raw scores match `DenseTaskRetriever` for identical encoder settings and fake embeddings.
- [ ] 6.5 Add task/content/node alignment validation before any ranking result is produced.
- [ ] 6.6 Run `uv run pytest tests/test_memory_stream_retrieval.py tests/test_batched_dense_encoding.py -q`.

## 7. Retrieval Registry and Runtime Provenance

- [ ] 7.1 Add `RetrievalMethodId.MEMORY_STREAM`, `MemoryStreamRetrievalSettings`, and `MemoryStreamBuildPayload` in `graph_memory/registry/retrieval.py`.
- [ ] 7.2 Extend `RetrievalJobSettings`, `RetrievalProvenance`, and retrieval builder registration for Memory Stream.
- [ ] 7.3 Add `RetrievalLifecycle.MEMORY_STREAM` and `ImportanceSource` to `graph_memory/registry/methods.py`; declare no graph/config/model artifact and an experiment-config dense encoder plus sidecar importance artifact.
- [ ] 7.4 Implement the Memory Stream builder in `graph_memory/registry/retrieval_builders.py` by reusing `_build_seed_retriever()` with dense settings.
- [ ] 7.5 Extend `RetrieveIO`, `run_retrieval.py`, and `graph_memory/stages/retrieve.py` to read and pass the importance artifact only for `MemoryStreamRetrievalSettings`.
- [ ] 7.6 Record importance path/model/prompt/content metadata, weights, recency decay, and effective encoder in retrieval provenance/run summary.
- [ ] 7.7 Add registry and stage tests for valid construction, missing artifact, task mismatch, injected fake encoder, and retrieval without importing or loading the local causal LLM.
- [ ] 7.8 Run `uv run pytest tests/test_memory_stream_retrieval.py tests/test_retrieval_registry_builders.py tests/test_config_run_retrieval.py tests/test_retrieval_provenance.py -q`.

## 8. Workflow Types, Artifacts, and Stage Config Compilation

- [x] 8.1 Add workflow `StageId.IMPORTANCE`, `WorkflowId.MEMORY_STREAM_RETRIEVAL`, and `ArtifactRole.IMPORTANCE_SCORES` in `scripts/workflow/types.py`, keeping it aligned with `graph_memory.registry.ids.StageId.IMPORTANCE`.
- [x] 8.2 Add `MEMORY_STREAM_WORKFLOW` with prepare, graphs, importance, retrieve, evaluate, and aggregate steps.
- [x] 8.3 Map the Memory Stream lifecycle to its workflow in `scripts/workflow/registry.py`.
- [x] 8.4 Allocate `runs/<name>/importance/test.memory_stream.importance.json` and its run summary in manifest artifacts.
- [x] 8.5 Extend `scripts/workflow/stage_configs.py` to compile importance and retrieve stage-root configs from the fixed experiment `memory_stream` section.
- [x] 8.6 Extend `CurrentWorkflowManifest` validation so `stage_configs.importance` exists and contains exactly methods whose workflows include the importance stage.
- [x] 8.7 Add `build_importance_commands()` and planner dispatch so workflow commands remain `script --config <stage-config>`.
- [x] 8.8 Run `uv run pytest tests/test_memory_stream_workflow.py tests/test_experiment_runner.py tests/test_workflow_orchestration.py -q`.

## 9. Planner Dependencies, Status, Resume, and Delivery

- [ ] 9.1 Add planner validation that retrieve-only Memory Stream execution requires a complete importance artifact when the importance stage is omitted.
- [ ] 9.2 Add importance status inspection using the sidecar plus matching successful run summary, expected task input, model, prompt version, generation settings, output path, and current ordered memory-item content digests.
- [ ] 9.3 Ensure cache-aware resume skips a complete importance stage and reruns a missing or stale one.
- [ ] 9.4 Update all closed stage-order assertions and range-selection tests for the new stage position.
- [ ] 9.5 Include the importance sidecar, annotation summary, and stage config in `scripts/deliver/collect_run_artifacts.py` while continuing to exclude the external cache directory.
- [ ] 9.6 Run `uv run pytest tests/test_memory_stream_workflow.py tests/test_experiment_runner.py tests/test_workflow_orchestration.py tests/test_deliver_run_artifacts.py -q`.

## 10. Experiment Configuration

- [ ] 10.1 Add `memory_stream` to the default experiment method list.
- [ ] 10.2 Add a complete fixed `memory_stream` section with semantic model id, local model path, prompt version, cache directory, device, trust-remote-code, low-CPU-memory, tensor-parallel-disabled, deterministic generation settings, equal weights, recency decay, and dense encoder settings.
- [ ] 10.3 Validate non-empty model id/path/prompt, supported device, positive max new tokens, `do_sample=false`, `tp_plan=null`, non-negative weights, at least one positive weight, and recency decay in `(0, 1]`.
- [ ] 10.4 Add smoke/quick/cloud profile behavior without generating train-split importance.
- [ ] 10.5 Add tests that selecting other methods never imports Transformers or requires the model path to exist and selecting Memory Stream fails on incomplete config.
- [ ] 10.6 Run `uv run pytest tests/test_memory_stream_workflow.py tests/test_experiment_runner.py -q`.

## 11. Documentation and MetaX Operations

- [ ] 11.1 Update retrieval/data/workflow contract docs with the sidecar schema, leakage boundary, pseudo-recency semantics, normalization, and latency boundary.
- [ ] 11.2 Add configuration documentation for every Memory Stream annotation and retrieval field.
- [ ] 11.3 Update operations commands with ModelScope and Hugging Face download commands for `Qwen/Qwen2.5-7B-Instruct` and the verified direct Transformers preflight script.
- [ ] 11.4 Document the required shell preparation: unset distributed rank/master variables, set `ACCELERATE_USE_DEEPSPEED=false`, select one card through `CUDA_VISIBLE_DEVICES`, and do not start vLLM or an HTTP server.
- [ ] 11.5 Document one long-lived annotation process, one model load per run with cache misses, no per-task process launch, annotation-only execution, restart behavior, retrieve-only execution, and full experiment execution.
- [ ] 11.6 Document that cache files are reusable local intermediates, final importance sidecars are reportable experiment artifacts, and annotation cost is reported separately from retrieval latency.

## 12. Verification

- [ ] 12.1 Run focused tests: `uv run pytest tests/test_memory_stream_importance.py tests/test_memory_stream_annotation_stage.py tests/test_memory_stream_retrieval.py tests/test_memory_stream_workflow.py -q`.
- [ ] 12.2 Run retrieval/workflow regression tests: `uv run pytest tests/test_batched_dense_encoding.py tests/test_retrieval_registry_builders.py tests/test_config_run_retrieval.py tests/test_experiment_runner.py tests/test_workflow_orchestration.py tests/test_deliver_run_artifacts.py -q`.
- [ ] 12.3 Run the full suite: `uv run pytest -q`.
- [ ] 12.4 Run `uv run ruff check .`.
- [ ] 12.5 Run `uv run basedpyright graph_memory scripts tests --level error`.
- [ ] 12.6 Run `uv run python scripts/experiment.py methods list` and verify `memory_stream` is present.
- [ ] 12.7 Run `uv run python scripts/experiment.py plan memory_stream_smoke --profile smoke --methods memory_stream --force --no-cache` and verify stage/config/artifact order.
- [ ] 12.8 Run `openspec validate add-memory-stream-retrieval --strict`.
- [ ] 12.9 Run `git diff --check`.
- [ ] 12.10 On the MetaX server, run a multi-task annotation smoke in one process and verify `model_load_count=1`; rerun it to verify all cache hits and `model_load_count=0`; then run retrieve/evaluate in an environment where Transformers/model files are unavailable and record the resulting summaries.

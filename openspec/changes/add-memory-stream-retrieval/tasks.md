## 1. Importance Contracts and Validation

- [x] 1.1 Define typed generation, task-record, final-artifact, annotation-result, and cache-stat contracts.
- [x] 1.2 Test exact producer task/node coverage, integer 1-10 validation, duplicate node rejection, content digests, and deterministic artifact order.
- [x] 1.3 Implement canonical memory-item serialization and SHA-256 digests excluding query, labels, runtime placement, and graph data.
- [x] 1.4 Export importance validators with task/node-specific failures.
- [x] 1.5 Run `uv run pytest tests/test_memory_stream_importance.py -q`.

## 2. Leakage-Safe Prompt and Response Parsing

- [x] 2.1 Prove query, answer, gold-node, and graph sentinels never enter prompt text or semantic cache keys.
- [x] 2.2 Implement `memory-stream-importance-v2` from ordered `{node_id, source, text, position}` records with ordered score-array output.
- [x] 2.3 Parse plain JSON and one optional Markdown JSON fence.
- [x] 2.4 Reject wrong-length arrays, booleans, floats, strings, and out-of-range values without fallback scores.
- [x] 2.5 Run the focused importance tests.

## 3. Persistent Local Transformers Runtime

- [x] 3.1 Prove all-cache-hit runs create zero runtimes and all misses share exactly one runtime.
- [x] 3.2 Implement lazy imports of Torch, AutoTokenizer, and AutoModelForCausalLM.
- [x] 3.3 Load with trusted remote code, automatic dtype, explicit one-device mapping, low-CPU-memory mode, and `tp_plan=None`.
- [x] 3.4 Implement deterministic chat-template generation, inference mode, token counts, and timings.
- [x] 3.5 Prove malformed output does not construct a second model instance.
- [x] 3.6 Prove annotation contains no HTTP, OpenAI SDK, vLLM, thread-pool generation, tensor parallelism, or per-task model loading.
- [x] 3.7 Run the focused importance tests.

## 4. Content-Addressed Cache and Atomic Writes

- [x] 4.1 Add and export `write_json_atomic()`.
- [x] 4.2 Store cache entries below `<cache_dir>/<digest-prefix>/<digest>.json`.
- [x] 4.3 Validate cache hits against model, prompt, generation, content digest, and exact nodes.
- [x] 4.4 Test cache hits, semantic invalidation, placement-only changes, corruption recovery, and failure restart.
- [x] 4.5 Run importance and IO observability tests.

## 5. Global Importance Prepare CLI

- [x] 5.1 Move `ImportanceAnnotationSettings` into the Memory Stream package rather than the workflow stage-config registry.
- [x] 5.2 Implement a standalone argparse CLI with zero-argument defaults for canonical dev tasks, global output, summary, cache, and local Qwen path.
- [x] 5.3 Clear distributed environment state before model-facing imports and preserve `CUDA_VISIBLE_DEVICES`.
- [x] 5.4 Validate canonical tasks, scan cache first, load one runtime only for misses, and atomically write the final artifact.
- [x] 5.5 Write success/failure summaries with task/item/cache/model/generation counts and timings.
- [x] 5.6 Derive the summary beside a custom output when `--summary` is omitted.
- [x] 5.7 Prove zero-argument smoke, cache-only rerun, failed-run preservation, and cross-platform summary paths.
- [x] 5.8 Implement subset selection by task id with duplicate, missing, digest, and node validation.
- [x] 5.9 Prove annotation is absent from workflow and stage-config registries.

## 6. Three-Signal Memory Stream Ranking

- [ ] 6.1 Add tests for raw recency, min-max normalization, constant signals, weighted sums, node-id ties, and empty retrieved edges.
- [ ] 6.2 Implement reusable min-max normalization with constant input mapped to zero.
- [ ] 6.3 Implement `MemoryStreamMethod` with an injected dense seed ranker, validated selected importance records, equal weights, and `recency_decay=0.99`.
- [ ] 6.4 Verify raw relevance matches `DenseTaskRetriever` for identical encoder settings.
- [ ] 6.5 Validate selected task/content/node alignment before ranking.
- [ ] 6.6 Run retrieval and dense regression tests.

## 7. Retrieval Registry and Runtime Provenance

- [ ] 7.1 Add `RetrievalMethodId.MEMORY_STREAM`, retrieval settings, and a build payload.
- [ ] 7.2 Extend retrieval job settings, provenance, and builder registration.
- [ ] 7.3 Declare dense encoder plus read-only global importance dependencies and no graph/config/checkpoint dependency.
- [ ] 7.4 Reuse the existing dense seed-retriever builder.
- [ ] 7.5 Extend retrieve IO and runtime loading only for Memory Stream.
- [ ] 7.6 Record global importance path/model/prompt/content metadata, weights, recency decay, and encoder.
- [ ] 7.7 Prove retrieval works without importing Transformers or accessing the local Qwen model.
- [ ] 7.8 Run registry, stage, and provenance tests.

## 8. Workflow External Dependency Integration

- [ ] 8.1 Add a Memory Stream workflow id without adding an importance stage or importance artifact role.
- [ ] 8.2 Use prepare, graphs, retrieve, evaluate, and aggregate steps.
- [ ] 8.3 Map the Memory Stream lifecycle to that workflow.
- [ ] 8.4 Compile only the retrieve stage config with the global importance path.
- [ ] 8.5 Keep annotation model/cache/generation settings out of experiment config and manifests.
- [ ] 8.6 Fail planning or retrieval when the global importance artifact is missing.
- [ ] 8.7 Prove plans contain no annotation command and manifests contain no run-local importance output.
- [ ] 8.8 Run workflow and experiment-runner tests.

## 9. Status, Resume, and Delivery

- [ ] 9.1 Keep workflow status free of an importance stage row.
- [ ] 9.2 Validate selected global records before retrieval rather than managing annotation resume in the workflow.
- [ ] 9.3 Record external importance provenance in retrieval summaries.
- [ ] 9.4 Keep the global artifact and cache outside run-local delivery ownership.
- [ ] 9.5 Run workflow status, resume, and delivery tests.

## 10. Experiment Configuration

- [ ] 10.1 Add `memory_stream` to the selected method list after retrieval implementation.
- [ ] 10.2 Add only retrieval weights, recency decay, dense encoder, and optional global importance path override.
- [ ] 10.3 Default the global path to `data/hotpotqa/processed/memory_stream/dev.importance.json`.
- [ ] 10.4 Validate non-negative weights, at least one positive weight, and recency decay in `(0, 1]`.
- [ ] 10.5 Prove other methods do not import Transformers or require the global artifact.
- [ ] 10.6 Run workflow and experiment-runner tests.

## 11. Documentation and MetaX Operations

- [ ] 11.1 Document the global sidecar schema, leakage boundary, subset selection, pseudo-recency, normalization, and latency boundary.
- [ ] 11.2 Document every retrieval field and the standalone CLI overrides.
- [ ] 11.3 Add ModelScope and Hugging Face download commands for Qwen2.5-7B-Instruct.
- [ ] 11.4 Document environment cleanup, one visible GPU, direct Transformers, and no HTTP/vLLM server.
- [ ] 11.5 Document `python scripts/annotate_importance.py`, restart behavior, and later retrieval consumption.
- [ ] 11.6 Document that cache and global artifacts are shared data resources, not workflow outputs.

## 12. Verification

- [ ] 12.1 Run focused importance prepare and retrieval tests.
- [ ] 12.2 Run retrieval/workflow regression tests.
- [ ] 12.3 Run `uv run pytest -q`.
- [ ] 12.4 Run `uv run ruff check .`.
- [ ] 12.5 Run `uv run basedpyright graph_memory scripts tests --level error`.
- [ ] 12.6 Run `uv run python scripts/annotate_importance.py --help`.
- [ ] 12.7 Verify the later Memory Stream plan contains no annotation stage.
- [ ] 12.8 Run `openspec validate add-memory-stream-retrieval --strict`.
- [ ] 12.9 Run `git diff --check`.
- [ ] 12.10 On MetaX, run the zero-argument command, verify one model load with misses, rerun for zero model loads, and retain the resulting global artifact for later retrieval work.

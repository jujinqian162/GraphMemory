## Why

Phase 2 requires a Generative Agents Memory Stream baseline, but the repository currently has no reproducible source for the query-independent `importance` signal and no method lifecycle that separates expensive local-LLM annotation from timed retrieval. The baseline must be implementable on the available MetaX C500 server without sending data to an external API or regenerating importance for every query.

## What Changes

- Add `memory_stream` as a public retrieval method implementing the simplified score `relevance + pseudo-recency + importance`.
- Add an offline importance-annotation stage that loads Qwen2.5-7B-Instruct directly through the working local `transformers` stack, validates 1-10 integer scores, and writes a leakage-safe sidecar artifact.
- Add content-addressed per-task caching and restart-safe assembly so repeated experiments reuse importance scores instead of rerunning the LLM.
- Keep one tokenizer/model instance resident for all cache misses in an annotation run; never load or unload the model per task.
- Keep importance query-independent: annotation receives memory item source/text/position only and never receives the retrieval query or gold labels.
- Derive pseudo-recency deterministically from existing item order because HotpotQA has no real timestamp or access history.
- Reuse the existing dense encoder for the relevance signal, normalize all three signals within each task, combine them with explicit weights, and preserve deterministic tie-breaking.
- Add workflow, manifest, stage-config, status/resume, experiment-config, run-summary, and operations documentation support for Memory Stream.
- Document direct single-process `AutoTokenizer`/`AutoModelForCausalLM` execution on MetaX, including distributed-environment cleanup, `CUDA_VISIBLE_DEVICES`, `tp_plan=None`, deterministic generation, and model load/generation timing.

## Capabilities

### New Capabilities

- `memory-stream-importance-annotation`: Leakage-safe local-LLM importance generation, schema validation, content-addressed caching, restart behavior, and reproducibility metadata.
- `memory-stream-retrieval`: Deterministic relevance, pseudo-recency, and importance normalization and ranking through the shared retrieval contracts.
- `memory-stream-experiment-workflow`: Method registration, stage planning, artifacts, status/resume, configuration, and operational commands for end-to-end experiments.

### Modified Capabilities

None.

## Impact

- Affected code: contracts and validation for importance artifacts, a persistent local-model runtime and annotation script, retrieval method settings/builders, retrieve-stage payloads, workflow stages/artifacts/manifests/status, experiment configuration, tests, and operations docs.
- Affected APIs: `RetrievalMethodId`, `RetrievalJobSettings`, method lifecycle metadata, stage config registry, workflow `StageId`/`ArtifactRole`, and retrieve-stage IO gain Memory Stream-specific variants or fields.
- Dependencies: annotation uses the vendor-compatible `torch` and `transformers` environment already proven on the MetaX server; imports are lazy so non-Memory-Stream workflows and unit tests do not load the LLM stack. No HTTP, vLLM, OpenAI SDK, or MetaX-specific Python API is introduced.
- Runtime: one annotation process owns one visible device and one resident Qwen model. Physical GPU selection is controlled before process startup through `CUDA_VISIBLE_DEVICES`.
- Artifacts: one final importance sidecar and run summary per Memory Stream experiment, plus reusable per-task cache entries outside timed retrieval output.

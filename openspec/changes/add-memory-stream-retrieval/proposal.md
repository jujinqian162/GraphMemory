## Why

Phase 2 requires a simplified Generative Agents Memory Stream baseline, but the
repository currently has no reproducible source for the query-independent
`importance` signal. Importance generation is expensive, independent of
workflow profiles, and reusable across all experiment workspaces. It must run
once as global preprocessing rather than being regenerated or owned by an
individual workflow run.

## What Changes

- Add a standalone global importance-preparation command that defaults to the
  canonical HotpotQA dev input and global data paths.
- Load Qwen2.5-7B-Instruct directly through the local `transformers` stack,
  validate exact 1-10 integer scores, and write a leakage-safe sidecar artifact.
- Add content-addressed per-task caching and restart-safe final assembly.
- Keep one tokenizer/model instance resident for all cache misses in one
  annotation process.
- Keep importance query-independent: prompts receive memory item
  source/text/position only and never receive queries or gold labels.
- Allow later workflow task subsets to select matching records from the complete
  global artifact by `task_id`, content digest, and exact node coverage.
- Add `memory_stream` retrieval later using normalized relevance,
  position-derived pseudo-recency, and the global offline importance input.
- Keep annotation outside workflow stages, run manifests, status/resume, and
  run-local artifact allocation.

## Capabilities

### New Capabilities

- `memory-stream-importance-annotation`: One-time global, leakage-safe local-LLM
  importance generation, strict validation, caching, restart behavior, and
  reproducibility metadata.
- `memory-stream-retrieval`: Deterministic relevance, pseudo-recency, and
  importance normalization and ranking.
- `memory-stream-experiment-workflow`: Method registration and consumption of a
  precomputed global importance dependency without owning its production.

### Modified Capabilities

None.

## Impact

- Affected code: importance contracts, validation, prompt construction, local
  runtime, cache, standalone CLI, later retrieval settings/builders, and
  workflow external-dependency checks.
- Affected APIs: the prepare milestone does not add a workflow stage or public
  retrieval method. Later milestones add Memory Stream retrieval variants and a
  read-only importance input to retrieval.
- Dependencies: annotation uses the vendor-compatible `torch` and
  `transformers` environment already proven on the MetaX server. No HTTP, vLLM,
  OpenAI SDK, or MetaX-specific Python API is introduced.
- Runtime: one standalone annotation process owns one visible device and one
  resident Qwen model.
- Artifacts: one global final importance sidecar and run summary for the
  canonical dev corpus, plus reusable per-task cache entries. Workflow runs
  reference this artifact but do not produce or copy it.

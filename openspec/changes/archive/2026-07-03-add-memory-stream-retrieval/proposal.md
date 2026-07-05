# Add Memory Stream Retrieval

## Why

The Phase 2 baseline matrix requires a simplified Memory Stream method using
`relevance + recency + importance`. HotpotQA supplies sentence order rather
than real timestamps, so recency is an explicit position-derived signal.

Importance labels already exist as an externally produced legacy artifact.
The repository needs deterministic cleaning, matching, normalization, and
read-only retrieval consumption, not an LLM annotation runtime.

## What Changes

- Add `scripts/data/clean_importance.py` to validate legacy labels against the
  canonical dev prefix and emit a compact schema-versioned artifact.
- Normalize unique score levels inside each task while preserving rank and ties.
- Keep strict task-id, content-digest, and node-coverage validation.
- Add Memory Stream retrieval using normalized relevance, pseudo-recency, and
  the cleaned importance sidecar.
- Load the cleaned sidecar once before timed retrieval, validate the requested
  task subset, and inject indexed scores into the method.
- Register Memory Stream as a stateless retrieval method and reuse the existing
  stateless retrieval workflow.
- Add an importance-backed prepare split source so a Memory Stream experiment
  can set `split_sources.dev/test = "importance"` and materialize run-local
  tasks from the cleaned sidecar's task ids instead of random sampling.
- Keep shared experiment splits as the default, but let the workflow cap the
  Memory Stream test split to the covered importance prefix and emit a warning
  instead of failing when the profile requests more examples than exist.
- Keep importance production outside workflow stages and run-local artifacts.
- Remove obsolete prompt, cache, model runtime, annotation CLI, and config.

## Capabilities

- `memory-stream-retrieval`: deterministic three-signal ranking.
- `memory-stream-experiment-workflow`: stateless-workflow integration with a
  read-only external importance dependency and a Memory Stream-only test split
  cap.

## Impact

- Affected code: importance contracts and validators, standalone data cleaning,
  retrieval settings/builders, workflow planning, provenance, tests, and docs.
- Artifacts: one compact normalized sidecar and one cleaning summary; no model
  cache or run-local importance output.

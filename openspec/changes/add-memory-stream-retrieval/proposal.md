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
- Keep importance production outside workflow stages and run-local artifacts.
- Remove obsolete prompt, cache, model runtime, annotation CLI, and config.

## Capabilities

- `memory-stream-retrieval`: deterministic three-signal ranking.
- `memory-stream-experiment-workflow`: workflow integration with a read-only
  external importance dependency.

## Impact

- Affected code: importance contracts and validators, standalone data cleaning,
  retrieval settings/builders, workflow planning, provenance, tests, and docs.
- Artifacts: one compact normalized sidecar and one cleaning summary; no model
  cache or run-local importance output.

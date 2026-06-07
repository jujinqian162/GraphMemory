## 1. Regression Tests

- [x] 1.1 Add parser/help tests proving `plan` and `run` expose `--no-cache` with default cache-aware wording.
- [x] 1.2 Add failing tests for typed command status keys and default completed-prefix pruning.
- [x] 1.3 Add failing tests proving `--no-cache` keeps the full selected plan for both plan rendering and run execution.
- [x] 1.4 Add failing tests for stale run-summary evidence stopping prefix pruning.
- [x] 1.5 Add status validation tests for non-retrieve stage summaries where the manifest and run summary disagree.

## 2. Resume Implementation

- [x] 2.1 Add typed status-key and cache-resume decision structures without introducing broad `Any` APIs.
- [x] 2.2 Implement completed-prefix pruning from live status rows.
- [x] 2.3 Wire default cache-aware pruning and `--no-cache` into `scripts/experiment.py plan` and `run`.
- [x] 2.4 Export the new runner helper through `scripts.workflow` and the compatibility facade only where useful.

## 3. Provenance Status

- [x] 3.1 Extend status inspection to validate prepare, graph, pair, train, tune, evaluate, and aggregate run summaries.
- [x] 3.2 Preserve existing retrieval stale detection and variant-qualified status behavior.
- [x] 3.3 Keep manifest `stage_status` JSON-compatible while using typed keys internally.

## 4. Documentation And Verification

- [x] 4.1 Update command documentation for default cache-aware resume and `--no-cache`.
- [x] 4.2 Run focused tests for workflow orchestration and CLI contracts.
- [x] 4.3 Run OpenSpec validation for `add-experiment-runner-cache-resume`.
- [x] 4.4 Run a smoke-profile workflow proving initial execution, cached resume, and no-cache behavior.
- [x] 4.5 Run full verification, commit, and push when all checks pass.

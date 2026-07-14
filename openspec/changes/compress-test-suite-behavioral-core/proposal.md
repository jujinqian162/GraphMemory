## Why

The test suite has grown to 471 collected cases across 69 test files, including configuration snapshots, migration tombstones, source-text assertions, and repeated checks of Pydantic or Hydra behavior. This obscures the tests that protect scientific correctness, slows routine verification, and makes harmless refactors require broad test maintenance.

## What Changes

- Replace the accumulated test inventory with a compact behavioral core of at most 100 collected pytest cases.
- Reduce the number of test files and total test lines by consolidating related scenarios around meaningful ownership boundaries.
- Remove tests that duplicate configuration values, third-party schema mechanics, deleted compatibility surfaces, exact internal exports, or source-code spellings.
- Preserve coverage for leakage prevention, dataset conversion, graph semantics, retrieval and evaluation metrics, trainable retrieval, beam decoding, checkpoint behavior, and representative experiment workflows.
- Define a durable admission rule for future tests so configuration-only edits and implementation-detail changes do not automatically add permanent tests.

## Capabilities

### New Capabilities

- `behavioral-test-suite`: Defines the compact, behavior-focused verification contract and the criteria for admitting or removing repository tests.

### Modified Capabilities


## Impact

- Primarily affects `tests/`, test fixtures/helpers, and `docs/30-design/testing-strategy.md`.
- Does not change production APIs, experiment semantics, model behavior, configuration values, or supported datasets and methods.
- Verification commands and contributor guidance will reference the compact suite and representative workflow smoke checks.

## 1. Baseline and Policy

- [x] 1.1 Record the current collected-case, test-file, and test-line baselines and identify behavior categories that must survive.
- [x] 1.2 Update the testing strategy with the behavioral admission rule and explicit non-goals for snapshot, tombstone, and third-party-mechanics tests.

## 2. Remove Low-Value Inventory

- [x] 2.1 Remove configuration-default snapshots, exact registry/export snapshots, Python metadata checks, and repeated Pydantic/Hydra mechanics tests.
- [x] 2.2 Remove migration tombstones, deleted-module assertions, source-token scans, and obsolete refactor-specific helpers.

## 3. Consolidate Behavioral Coverage

- [x] 3.1 Consolidate dataset conversion, leakage, graph, retrieval, and evaluation coverage around representative scientific scenarios.
- [x] 3.2 Consolidate train-pair, tensorization, R-GCN, beam, Dense-FT, checkpoint, and inference coverage around representative model scenarios.
- [x] 3.3 Consolidate experiment composition, planning, resume, tracking, stage execution, and artifact coverage around representative workflow scenarios.

## 4. Verify Budgets and Behavior

- [x] 4.1 Confirm no more than 100 pytest cases are collected and record the reduced test-file and test-line counts.
- [x] 4.2 Run the compact pytest suite and repair any failures without restoring low-value snapshot tests.
- [x] 4.3 Run Ruff, BasedPyright, compileall, Git diff checks, and strict OpenSpec validation.
- [x] 4.4 Run representative supported-dataset workflow smoke checks not already exercised by the compact suite.

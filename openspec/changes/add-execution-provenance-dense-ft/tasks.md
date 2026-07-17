## 1. Contract Tests

- [x] 1.1 Update Registry tests to require `dense_ft` in the execution-provenance family while preserving its flat request, artifact, and capability contract.
- [x] 1.2 Add workflow-plan tests proving provenance Dense-FT has text-only pairs with effective graph-neighbor sampling set to zero and no EvidenceGraph stage/dependency.
- [x] 1.3 Add pair-stage coverage proving graphless Dense-FT tasks build valid text-only pairs while evidence-family Dense-FT retains graph inputs and configured graph-neighbor sampling.

## 2. Dense-FT Provenance Support

- [x] 2.1 Widen the existing `dense_ft` Registry definition to support execution-provenance requests without changing public method IDs or retrieval settings.
- [x] 2.2 Make workflow pair planning family-aware, omit provenance EvidenceGraph artifacts, and emit a provenance Dense-FT sampling config with graph-neighbor negatives disabled.
- [x] 2.3 Generalize train-pair construction to attach optional graphs by task instead of dispatching on the provenance R-GCN method name.

## 3. User-Facing Workflow

- [x] 3.1 Update the execution-provenance runbook method matrix and copy-pastable commands to include `dense_ft` and explain that it remains a graph-free supervised text baseline.

## 4. Verification

- [x] 4.1 Run focused Registry, pair-building, Dense-FT workflow, and `twowiki_provenance` tests.
- [x] 4.2 Run a one-example CPU `twowiki_provenance` Dense-FT smoke plan/workflow and verify pair, model, prediction, metric, and aggregate artifacts with no EvidenceGraph stage.
- [x] 4.3 Run Ruff, basedpyright, compileall, strict OpenSpec validation, and `git diff --check`.

## 5. Tracking Regression Fix

- [x] 5.1 Add a failing `execute_experiment` regression proving provenance Dense-FT pair and train stages log one stable effective graph-neighbor sampling value.
- [x] 5.2 Centralize provenance Dense-FT effective method-config projection across pair/train planning and rerun focused tracking/workflow plus quality gates.

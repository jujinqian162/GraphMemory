## Why

The execution-provenance benchmark already exposes dataset-owned flat text requests for BM25 and Dense, but `dense_ft` is artificially restricted to evidence datasets and its pair-planning path assumes an EvidenceGraph that this dataset intentionally does not provide. This prevents a supervised in-domain text baseline from running even though the Dense-FT trainer, labels, and candidate projection are otherwise compatible.

## What Changes

- Allow the existing public `dense_ft` method to train, retrieve, evaluate, and aggregate on execution-provenance datasets through their dataset-owned `TextRankingRequest` projection.
- Make pair planning task-family-aware so execution-provenance Dense-FT does not request an EvidenceGraph and builds text-only negatives.
- Force the effective provenance Dense-FT pair-stage configuration to disable graph-neighbor negatives because no EvidenceGraph is available, while leaving evidence-dataset Dense-FT defaults unchanged.
- Carry that effective sampling policy consistently through pair and train stage configs so MLflow sees one immutable parameter set for the Dense-FT baseline.
- Extend config composition, workflow validation, focused tests, and the execution-provenance runbook for the new method/dataset combination.
- Keep `execution_provenance_rgcn_retriever` Dense-seeded and unchanged; do not add a Dense-FT-seeded provenance R-GCN method in this change.

## Capabilities

### New Capabilities

- `execution-provenance-dense-ft-retrieval`: Defines supervised flat Dense-FT training and retrieval on execution-provenance text candidates.

### Modified Capabilities

- `semantic-method-registry`: Widens the existing `dense_ft` method to the execution-provenance task family without changing its `TextRankingRequest` or model-directory contract.
- `retrieval-workflow-matrix`: Adds `dense_ft` to the execution-provenance method matrix and defines a text-only pair/train/retrieve/evaluate workflow with no EvidenceGraph stage.

## Impact

Affected surfaces include Registry task-family metadata, experiment configuration validation, effective method-config projection across pair/train planning, text-only pair construction, execution-provenance dataset method defaults/runbook, MLflow tracking, and focused registry/workflow/smoke tests. Public method IDs, dataset schemas, generated raw artifacts, Dense-FT checkpoints, evidence R-GCN methods, and both provenance retrievers remain unchanged.

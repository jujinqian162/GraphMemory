## Why

MuSiQue is the next hard multi-hop dataset target for this project, and the current dataset selector only supports HotpotQA and 2Wiki. Adding MuSiQue-Ans lets the existing evidence retrieval, graph rerank, Dense-FT, and R-GCN workflows run on 2-4 hop paragraph-level supervision without changing reusable retrieval contracts.

## What Changes

- Add a MuSiQue dataset package with strict raw parsing, leakage-safe ranking/label conversion, request projectors, and validation.
- Add `scripts/prepare_musique.py` to convert official MuSiQue-Ans JSONL files into `*.input.json` and `*.labels.json` artifacts.
- Extend dataset dispatch, direct CLI choices, and workflow prepare-stage routing to accept `dataset: "musique"`.
- Add a usable experiment config for MuSiQue evidence retrieval with smoke/quick/full profiles and current baseline/trainable methods.
- Add focused tests for parsing, conversion, leakage boundaries, selector dispatch, workflow wiring, and config loading.

## Capabilities

### New Capabilities
- `musique-dataset`: MuSiQue-Ans paragraph-level examples can be prepared and consumed by the existing request-first evidence retrieval workflow.

### Modified Capabilities

None.

## Impact

- Affected production code:
  - `graph_memory/datasets/`
  - `graph_memory/validation/`
  - `scripts/prepare_musique.py`
  - dataset-aware CLI/workflow files under `scripts/` and `graph_memory/registry/`
  - `configs/experiments/`
- No external Python dependency changes are expected.
- No answer generation or MuSiQue-Full answerability/sufficiency evaluation is included in this change.

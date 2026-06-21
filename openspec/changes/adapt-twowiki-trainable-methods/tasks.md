## 1. 2Wiki Trainable Configuration

- [x] 1.1 Update `configs/experiments/2wiki_tiny.json` so `methods` includes `dense_ft` and `dense_rgcn_graph_retriever`.
- [x] 1.2 Add `method_configs` to `configs/experiments/2wiki_tiny.json` for `dense_ft` and `dense_rgcn_graph_retriever`.
- [x] 1.3 Remove the Dense-FT smoke CPU exception by ensuring the method config used by 2Wiki smoke resolves Dense-FT trainer device to `cuda`.
- [x] 1.4 Confirm the R-GCN method config used by 2Wiki smoke resolves train and retrieve device to `cuda`.
- [x] 1.5 Keep 2Wiki graph configuration on the existing visible edge vocabulary without adding 2Wiki relation edge types.

## 2. Workflow and Stage Config Tests

- [x] 2.1 Extend 2Wiki workflow tests so the named `2wiki_tiny` config initializes `dense_ft` and `dense_rgcn_graph_retriever`.
- [x] 2.2 Add assertions that generated `pairs`, `train`, `retrieve`, and `evaluate` stage configs for both trainable methods use `dataset: "twowiki"`.
- [x] 2.3 Add assertions that Dense-FT train and retrieve stage configs use `device: "cuda"`.
- [x] 2.4 Add assertions that R-GCN train and retrieve stage configs use `device: "cuda"`.
- [x] 2.5 Add assertions that Dense-FT retrieve stage configs do not require graph input and R-GCN train/retrieve stage configs do require graph artifacts.

## 3. Supervision, Leakage, and Metric Semantics

- [x] 3.1 Add or extend train-pair tests proving 2Wiki positives come from `EvidenceLabel.gold_evidence_item_ids`.
- [x] 3.2 Add or extend leakage tests proving `gold_dependency_edges`, `supporting_facts`, `evidences`, `evidences_id`, and `answer` do not enter `GraphBuildRequest.input_visible_edges` or graph artifacts.
- [x] 3.3 Add or extend evaluation tests proving Dense-FT 2Wiki path metrics are `N/A`.
- [x] 3.4 Add or extend evaluation tests proving R-GCN 2Wiki path metrics are numeric when evaluated labels include non-empty `gold_dependency_edges`.
- [x] 3.5 Add or extend evaluation tests proving R-GCN missed path coverage contributes `0.0` rather than `N/A`.

## 4. Verification

- [x] 4.1 Run focused tests for 2Wiki workflow, trainable configs, leakage boundaries, and path metrics.
- [x] 4.2 Run trainable regression tests covering Dense-FT workflow and R-GCN workflow/stage config behavior.
- [x] 4.3 Run `uv run python scripts/experiment.py plan 2wiki_trainable_smoke --config configs/experiments/2wiki_tiny.json --profile smoke --methods dense_ft,dense_rgcn_graph_retriever --force --no-cache`.
- [x] 4.4 Run `uv run python scripts/experiment.py run 2wiki_trainable_smoke --config configs/experiments/2wiki_tiny.json --profile smoke --methods dense_ft,dense_rgcn_graph_retriever --force`.
- [x] 4.5 Run `uv run python scripts/experiment.py status 2wiki_trainable_smoke` and verify predictions plus `main_results.csv`, `path_results.csv`, and `efficiency_results.csv` exist.
- [ ] 4.6 Run static and repository-level gates appropriate for this change, including `uv run pytest -q`, `uv run ruff check .`, `uv run basedpyright`, `uv run python -m compileall graph_memory scripts tests`, and `git diff --check`.
- [x] 4.7 Run `openspec validate adapt-twowiki-trainable-methods --strict`.

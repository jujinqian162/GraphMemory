## 1. Registry and Method Config

- [x] 1.1 Add failing registry/config tests for `learned_graph_rgcn_retriever`, path-metric support, and required loss config validation.
- [x] 1.2 Add `RetrievalMethodId.LEARNED_GRAPH_RGCN_RETRIEVER` and register the method as checkpoint-backed graph-aware R-GCN trainable.
- [x] 1.3 Add learned graph R-GCN method config records, including proposal graph settings and rank/edge/sparse loss settings with defaults `1.0/0.2/0.05`.
- [x] 1.4 Add `configs/methods/learned_graph_rgcn_retriever.json`.

## 2. Proposal Graph Stage

- [x] 2.1 Add failing workflow/proposal graph tests proving learned method uses method-local proposal graph artifacts and does not depend on shared graph artifacts.
- [x] 2.2 Add proposal graph build config and CLI that reuse dataset selectors and existing `GraphBuilder` rules with higher recall caps.
- [x] 2.3 Wire manifest, stage config, and planner support for `proposal_graphs` in the learned method dependency chain.

## 3. Learned Edge Batching

- [x] 3.1 Add failing batching tests for HotpotQA empty edge labels and 2Wiki positive edge-label matching.
- [x] 3.2 Extend graph retriever batch contracts with optional learned edge tensors while preserving existing R-GCN batches.
- [x] 3.3 Build learned edge features and edge label masks from proposal graph edges and label-side `gold_dependency_edges`.

## 4. Edge Gate Model

- [x] 4.1 Add failing model tests for gate dimensions and zero-gate message suppression.
- [x] 4.2 Add `EdgeGateScorer` and learned graph model output carrying node logits, edge gate logits, and edge gates.
- [x] 4.3 Add gated R-GCN path that scales proposal edge weights by learned gates and factory support for the learned model.

## 5. Rank Edge Sparse Training Loss

- [x] 5.1 Add failing training-loss tests for default weights, weighted total loss, no-label edge loss, and zero edge-loss weight.
- [x] 5.2 Compute rank, edge, and sparse losses in learned graph training while preserving existing R-GCN training.
- [x] 5.3 Record loss components, sample counts, mean edge gate, and loss weights in train/dev metrics and checkpoint metadata.

## 6. Checkpoint and Inference

- [x] 6.1 Add failing checkpoint/retrieval tests for learned method provenance and method mismatch rejection.
- [x] 6.2 Save/load learned graph model config, loss config, and method identity through existing checkpoint-backed graph retrieval.
- [x] 6.3 Ensure learned retrieval predictions include ranked nodes and `retrieved_subgraph` derived from the proposal graph.

## 7. HotpotQA Workflow Wiring

- [x] 7.1 Add failing HotpotQA plan/config tests for learned method stage order and method config mapping.
- [x] 7.2 Add `learned_graph_rgcn_retriever` to active HotpotQA experiment configs without changing existing method behavior.
- [x] 7.3 Update method docs/commands for the new method.

## 8. 2Wiki Workflow and Leakage

- [x] 8.1 Add failing 2Wiki plan/leakage/edge-loss tests for learned method selection, gold-field exclusion, and positive edge labels.
- [x] 8.2 Add `learned_graph_rgcn_retriever` to active 2Wiki experiment configs without leaking gold fields into proposal graphs.
- [x] 8.3 Confirm path metrics continue to use `retrieved_subgraph` plus label-side `gold_dependency_edges`.

## 9. Verification

- [x] 9.1 Run focused learned graph R-GCN tests outside the Windows sandbox.
- [x] 9.2 Run `uv run basedpyright --level error` outside the Windows sandbox.
- [x] 9.3 Run `uv run pytest -q` outside the Windows sandbox.
- [x] 9.4 Run HotpotQA and 2Wiki plan smokes for `learned_graph_rgcn_retriever`.
- [x] 9.5 Run `openspec validate add-learned-graph-rgcn-retriever --strict`.

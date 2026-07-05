## 1. Workflow Contract Tests

- [x] 1.1 Add a failing test proving `learned_graph_rgcn_retriever` expands `dense_ft` as a pair/train dependency without adding `dense_ft` to selected retrieve/evaluate outputs.
- [x] 1.2 Add a failing test proving the learned graph train stage config sets `io.seed_checkpoint` to `learned/dense_ft/checkpoints/best_model`.
- [x] 1.3 Add or update a failing test proving learned graph pair/train/retrieve/evaluate configs still read method-local proposal graph artifacts.

## 2. Workflow Implementation

- [x] 2.1 Update the method registry so `learned_graph_rgcn_retriever` declares `seed_method=dense_ft` and `train_dependencies=(dense_ft,)`.
- [x] 2.2 Verify existing stage config generation emits `io.seed_checkpoint` for `learned_graph_rgcn_retriever` through the shared seed-checkpoint path.
- [x] 2.3 Verify selected methods, prediction paths, metrics paths, and aggregate outputs still expose only explicitly selected methods.

## 3. Documentation

- [x] 3.1 Update method config docs for `learned_graph_rgcn_retriever` to describe the implicit Dense-FT seed encoder dependency.
- [x] 3.2 Update experiment config docs where trainable method dependencies are described.
- [x] 3.3 Update the learned graph plan notes to record that the current default learned graph method is Dense-FT seeded.

## 4. Verification

- [x] 4.1 Run targeted workflow tests covering learned graph proposal graphs and 2Wiki trainable stage configs.
- [x] 4.2 Run current method/config contract tests impacted by train dependency expansion.
- [x] 4.3 Run type/lint checks required by the repository workflow.

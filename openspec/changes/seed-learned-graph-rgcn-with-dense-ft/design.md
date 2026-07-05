## Context

`learned_graph_rgcn_retriever` currently uses method-local proposal graphs and learned edge gates, but its R-GCN encoder is still seeded from the base dense encoder configured in `configs/methods/learned_graph_rgcn_retriever.json`. Recent 2Wiki results show that this method improves graph/path structure over `dense_rgcn_graph_retriever`, while `dense_ft_rgcn_graph_retriever` remains stronger on final evidence ranking because it uses the run-local Dense-FT checkpoint as its seed/text encoder.

The codebase already has the required seed mechanism: R-GCN train stage configs can carry `io.seed_checkpoint`, and `RgcnGraphRetrieverTrainer` resolves that checkpoint into effective encoder settings through Dense-FT model metadata. The workflow registry also already models train dependencies through `MethodDefinition.train_dependencies`.

## Goals / Non-Goals

**Goals:**

- Make `learned_graph_rgcn_retriever` use the run-local Dense-FT checkpoint as its seed encoder.
- Keep `learned_graph_rgcn_retriever` as the only public learned graph method; no new method id, config file, or result-table row.
- Preserve method-local proposal graph artifacts for `learned_graph_rgcn_retriever`.
- Reuse the existing `seed_checkpoint` and Dense-FT metadata path instead of introducing a new encoder override.
- Keep existing rank/edge/sparse loss behavior unchanged.

**Non-Goals:**

- Do not remove `dense_ft_rgcn_graph_retriever`.
- Do not make proposal graphs depend on the shared `graphs` stage.
- Do not hard-code a previous run's Dense-FT checkpoint path into method config.
- Do not tune loss weights as part of this change.

## Decisions

1. Use workflow dependency metadata, not method config paths.

   `learned_graph_rgcn_retriever` will declare `seed_method=RetrievalMethodId.DENSE_FT` and `train_dependencies=(RetrievalMethodId.DENSE_FT,)` in the method registry. This keeps the checkpoint path run-local and lets existing manifest/stage-config code derive `learned/dense_ft/checkpoints/best_model`.

   Alternative considered: set `encoder.model_name` in `configs/methods/learned_graph_rgcn_retriever.json` to a Dense-FT output path. That would be brittle because the path is run-specific and would not train the dependency automatically.

2. Reuse existing R-GCN `seed_checkpoint` behavior.

   `scripts/workflow/stage_configs.py` already emits `io.seed_checkpoint` when a method's `seed_method` is `dense_ft`. `graph_memory/stages/trainers.py` already loads Dense-FT metadata and replaces the effective encoder model with that checkpoint directory. The implementation should avoid duplicating this path.

   Alternative considered: add learned-graph-specific trainer logic. That would create a second seed-encoder path for the same behavior and increase drift risk.

3. Keep proposal graph ownership unchanged.

   `learned_graph_rgcn_retriever` will continue to use `proposal_graphs/<method>` artifacts for pairs, train, retrieve, and evaluate. The new Dense-FT dependency only supplies encoder weights and does not change graph construction inputs.

   Alternative considered: reuse the Dense-FT seeded R-GCN shared graph workflow. That would discard the learned proposal/gated graph behavior this method is meant to test.

## Risks / Trade-offs

- Dense-FT training becomes an implicit cost for `learned_graph_rgcn_retriever` runs -> The manifest and stage plan must show `dense_ft` pair/train commands so the cost is visible before execution.
- Existing docs may still describe learned graph as base-E5 seeded -> Update method and experiment docs where they describe trainable method dependencies.
- Changing the default method behavior affects comparability with previous `learned_graph_rgcn_retriever` runs -> The OpenSpec and docs must state that new runs use Dense-FT seed encoder by default; previous result directories keep their own provenance.
- Dependency expansion could accidentally add `dense_ft` to retrieve/evaluate outputs -> Tests must assert `dense_ft` is only added to pair/train dependency execution when the user selected only `learned_graph_rgcn_retriever`.

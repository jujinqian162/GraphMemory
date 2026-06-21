## Context

The current 2Wiki path has dataset-specific parsing, conversion, projectors, validation, path metrics, and a named `2wiki_tiny` experiment config. The remaining gap is that the named 2Wiki workflow only selects `bm25`, `dense`, and `dense_graph_rerank`, so it does not yet exercise the Phase 2 trainable methods: `dense_ft` and `dense_rgcn_graph_retriever`.

The runtime already has the right request-first boundary. Dataset records project into `TextRankingRequest`, `GraphBuildRequest`, and `EvidenceEvaluationRequest`; trainable training loads those requests through dataset selectors. R-GCN consumes graph artifacts and returns graph-aware `retrieved_subgraph` edges. Dense-FT remains a flat text retriever and does not consume graph artifacts during retrieval.

The user constraint for this change is explicit: all trainable devices must resolve to `cuda`. This means the implementation must not add a CPU-only smoke path, and any Dense-FT smoke profile used by the 2Wiki trainable workflow must not resolve `device` to `cpu`.

## Goals / Non-Goals

**Goals:**

- Make `dense_ft` and `dense_rgcn_graph_retriever` first-class methods in the named 2Wiki experiment flow.
- Ensure every generated trainable train/retrieve stage config for this flow uses `device: "cuda"`.
- Reuse existing 2Wiki `supporting_facts` as node-level supervision for both trainable methods.
- Keep `gold_dependency_edges` label-only and use them only for path metrics.
- Preserve Dense-FT as a flat baseline with `Path Recall@10` and `Edge Recall@10` reported as `N/A`.
- Preserve R-GCN as graph-aware Ours with path metrics computed from returned `retrieved_subgraph`.
- Verify the workflow through OpenSpec tasks and real experiment runner commands.

**Non-Goals:**

- Do not add new R-GCN edge types or relation vocab entries for 2Wiki.
- Do not train R-GCN with edge/path loss in this change.
- Do not add `dense_ft_graph_rerank` or another hybrid method in this change.
- Do not leak 2Wiki `supporting_facts`, `evidences`, `evidences_id`, `answer`, or `gold_dependency_edges` into test-time graph construction.
- Do not introduce CPU fallback behavior for trainable 2Wiki smoke verification.

## Decisions

### Decision 1: Extend the named 2Wiki experiment rather than adding a new workflow family

The existing workflow registry already maps method lifecycles to the correct stages. `dense_ft` gets pairs, train, retrieve, evaluate, and aggregate stages. `dense_rgcn_graph_retriever` gets pairs, train, graph-backed retrieve, evaluate, and aggregate stages. The lowest-risk path is to add method selection and method config wiring to the 2Wiki named config and prove the generated stage configs are dataset-aware.

Alternative considered: add a separate 2Wiki trainable workflow. That would duplicate lifecycle logic already represented by the method registry and workflow registry.

### Decision 2: Keep training supervision node-level for both methods

Dense-FT and R-GCN should train from 2Wiki `gold_evidence_sentence_ids`, which come from `supporting_facts`. This matches the current train-pair builder and both trainable method contracts. `gold_dependency_edges` stay out of training loss for this change.

Alternative considered: make R-GCN path-aware by training on `gold_dependency_edges`. That would require a new pair/path training contract and risks mixing label-only edges into model-visible graph inputs.

### Decision 3: Keep Dense-FT path metrics unsupported

Dense-FT produces ranked sentence nodes without meaningful retrieved graph edges. It should continue to report path metrics as `N/A`. This makes it a node-retrieval baseline against which R-GCN can be compared on node metrics, while graph-aware methods own path metrics.

Alternative considered: make Dense-FT emit an induced graph so it can receive numeric path metrics. That would blur the baseline and create a hidden graph method without a method name or lifecycle change.

### Decision 4: Enforce CUDA at config-resolution boundaries

The implementation should make generated train/retrieve stage configs for `dense_ft` and `dense_rgcn_graph_retriever` resolve `device` to `cuda`. Tests should inspect generated stage configs rather than relying only on source JSON, because method profiles and experiment profiles are merged before execution.

Alternative considered: rely on existing method config defaults. That is insufficient because current Dense-FT smoke behavior can resolve to CPU.

### Decision 5: Keep 2Wiki graph construction vocabulary unchanged

Visible graphs should continue using the current edge vocabulary and graph construction rules. 2Wiki `evidences` and `evidences_id` only create label-side dependency edges for evaluation.

Alternative considered: add relation-aware visible edges from 2Wiki triples. That is out of scope because supervised triples are label annotations, and adding unsupervised relation extraction would require new extraction, tensorization, config, ablation, and evaluation contracts.

## Risks / Trade-offs

- Dense-FT smoke may become GPU-only if the shared Dense-FT smoke profile is changed to `cuda` -> Mitigate by documenting that this change intentionally follows the CUDA-only constraint and by keeping any future local CPU profile separate.
- 2Wiki path-supported sample count may vary by split and question type -> Mitigate by recording path-supported counts in preparation summaries and only computing path metrics over labels with non-empty dependency edges.
- R-GCN path metrics may be low even when node recall improves -> Mitigate by keeping node metrics and path metrics separate in result tables and failure analysis.
- Existing HotpotQA workflows could be affected if shared method configs are changed -> Mitigate with focused HotpotQA workflow/config regression tests or use dedicated CUDA method config files if preserving global smoke semantics becomes necessary.
- Full trainable smoke can be expensive on GPU -> Mitigate by keeping `2wiki_tiny` smoke splits small while still running real train/retrieve/evaluate stages.

## Migration Plan

1. Update 2Wiki experiment configuration to include trainable methods and method configs.
2. Ensure Dense-FT and R-GCN trainable configs used by this experiment resolve every train/retrieve device to `cuda`.
3. Add tests for 2Wiki trainable manifest generation, stage config device values, and path metric support boundaries.
4. Run focused tests and a tiny 2Wiki trainable experiment smoke through `scripts/experiment.py`.
5. Stop before broader quick/full runs until smoke artifacts and table semantics are reviewed.

## Open Questions

- None for the OpenSpec proposal. If later local CPU verification is needed, it should be handled as a separate profile decision rather than weakening this change's CUDA-only constraint.

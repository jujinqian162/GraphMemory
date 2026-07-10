## Context

The repository already has a request-first cross-dataset boundary: dataset-specific raw records are converted into ranking and label artifacts, then projected into dataset-neutral retrieval, graph build, and evaluation requests. HotpotQA and 2Wiki are implemented behind `graph_memory/datasets/<dataset>/`, with dataset selection centralized in `graph_memory/datasets/selection.py` and workflow routing in `scripts/workflow/`.

MuSiQue official data is JSONL. For MuSiQue-Ans, every example is answerable and contains paragraph-level evidence labels. The useful official fields for this project are:

- `id`
- `question`
- `answer`
- `answer_aliases`
- `answerable`
- `paragraphs[]`: `idx`, `title`, `paragraph_text`, `is_supporting`
- `question_decomposition[]`: `id`, `question`, `answer`, `paragraph_support_idx`

## Goals / Non-Goals

**Goals:**

- Support MuSiQue-Ans as `dataset: "musique"` in the existing evidence retrieval workflow.
- Preserve the existing leakage boundary: retrieval inputs include question and candidate paragraphs, labels include answer/support/dependency supervision.
- Emit paragraph-level gold evidence IDs and dependency edges usable by existing recall and path metrics.
- Provide a ready-to-use experiment config with smoke/quick/full profiles and current method set.
- Document the official dataset download command and expected raw file paths.

**Non-Goals:**

- Do not add answer generation.
- Do not implement MuSiQue-Full unanswerable/sufficiency evaluation.
- Do not introduce sentence splitting or chunking in the first version.
- Do not change reusable retrieval, graph, or evaluation request contracts.

## Decisions

1. **Represent each MuSiQue paragraph as one evidence item.**

   Candidate IDs will be deterministic paragraph node IDs derived from official paragraph indices, e.g. `p0`, `p1`. Text ranking candidates will use `"<title>. <paragraph_text>"`; graph nodes will use `paragraph_text` with `source_ref=<title>`, `group_key="document:<title>"`, and `sequence_index=<idx>`.

   Alternative considered: sentence-split paragraphs to match HotpotQA. Rejected because official MuSiQue support labels are paragraph indices; splitting would require lossy label expansion and make support metrics less faithful.

2. **Construct gold evidence from `paragraphs[].is_supporting`.**

   `is_supporting` is the official support label used by MuSiQue evaluation. The converter will map each supporting paragraph's official `idx` to the corresponding `p{idx}` node ID.

   Alternative considered: use only `question_decomposition[].paragraph_support_idx`. Rejected because official support metrics are based on `paragraphs[].is_supporting`, while decomposition support is better treated as path supervision.

3. **Construct dependency edges from decomposition references.**

   Each decomposition step with a `paragraph_support_idx` maps to `p{idx}`. If a step question references prior steps using `#N`, the converter adds edges from the referenced prior step's support paragraph to the current step's support paragraph. Duplicate/self edges are removed. The label metadata records `path_supported` and the number of unmapped or out-of-range references.

   Alternative considered: use decomposition order as a simple chain. Rejected because MuSiQue decomposition questions explicitly encode dependencies via `#N`; order-only chains can create false edges.

4. **Prepare only answerable examples by default.**

   `prepare_musique.py` will accept official MuSiQue-Ans files and require `answerable == true` unless invalid examples are being dropped. This keeps the first config aligned with retrieval-only evaluation.

   Alternative considered: accept MuSiQue-Full now. Rejected because answerability/sufficiency metrics add a different task and would blur this change's evidence retrieval boundary.

5. **Mirror 2Wiki workflow integration.**

   MuSiQue will add the same public surfaces as 2Wiki: dataset package exports, validation functions, dataset selector dispatch, direct script dataset choices, workflow prepare routing, named config, and tests.

## Risks / Trade-offs

- [Long paragraph truncation in dense encoders] → Keep paragraph-level labels for correctness; expose smoke/quick profiles and leave chunking as a future measured change if token-length analysis proves necessary.
- [Sparse graph structure] → Paragraph nodes have fewer within-document sequence edges than sentence nodes; graph construction still receives title/group/sequence metadata and can use existing lexical/entity bridge logic.
- [Malformed decomposition references] → Parser/converter fail fast in strict mode and drop/count invalid examples in non-strict prepare mode.
- [Metric comparability] → MuSiQue recall is paragraph-level and should not be compared numerically as sentence-level HotpotQA/2Wiki recall without noting granularity.

## Migration Plan

1. Add MuSiQue parser/converter/projector/validation and focused tests.
2. Extend dataset dispatch and CLI/workflow dataset choices.
3. Add `scripts/prepare_musique.py` and config `configs/experiments/musique_evidence_retrieval.json`.
4. Verify smoke-level prepare/workflow planning and focused unit tests.

Rollback is straightforward: remove the MuSiQue package, script, config, tests, and selector/workflow dispatch branches.

## Open Questions

None for the first version. The deliberate boundary is MuSiQue-Ans paragraph-level evidence retrieval only.

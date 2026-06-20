## Context

The request-first HotpotQA refactor introduced consumer-owned request contracts, but the current runtime still lets HotpotQA record semantics cross the projector boundary. The clearest example is `GraphEvidenceNode`: it is named as a general graph-domain node, yet its fields are HotpotQA/document-sentence shaped (`document_sentence`, `source`, `sentence_id`, `position`). The same pattern appears in retrieval execution, ranked-result assembly, training-pair construction, Dense-FT data, R-GCN batching/training/inference, evaluation/failure cases, validation, Memory Stream importance artifacts, and tuning helpers.

The architectural rule for this change is stricter than the first request-first refactor: after a dataset record has been projected to a consumer request, reusable domain/model code must not import dataset records, call dataset projectors, or require dataset-specific field names. Dataset-specific records remain valid at dataset, script, application, and stage boundaries where file loading and projection are explicit.

## Goals / Non-Goals

**Goals:**

- Make `MemoryGraph` use dataset-neutral graph node semantics after `GraphBuildRequest` enters graph/domain code.
- Ensure reusable retrieval, training, model, evaluation, validation, and Memory Stream internals consume request-level or domain-level task specs rather than `HotpotQARankingRecord` / `HotpotQALabelRecord`.
- Remove reverse projection from domain requests back into HotpotQA records, especially in R-GCN inference.
- Keep HotpotQA projectors in `graph_memory.datasets.hotpotqa.projectors` and move any remaining HotpotQA-aware assembly to script/stage/application edges.
- Add architecture tests and durable docs that make the dataset/domain boundary visible and enforceable.

**Non-Goals:**

- Do not add support for a new dataset in this change.
- Do not change public CLI flags, file names, retrieval method names, config names, ranking formulas, model math, metric formulas, or train-pair sampling ratios.
- Do not remove HotpotQA prepared artifact contracts; they remain the current dataset-owned artifacts.
- Do not introduce a generic plugin system, dependency injection container, or broad compatibility facade.
- Do not redesign graph-rerank edge scoring or R-GCN neural architecture.

## Decisions

### Decision: Domain graph nodes are dataset-neutral graph items

`MemoryGraph` will keep a dedicated query node, but non-query nodes will use abstract graph-item semantics rather than HotpotQA sentence semantics. A graph item should carry stable fields such as `id`, `node_kind`, `text`, `source_ref`, `group_key`, `sequence_index`, and `metadata`. The exact type name can be chosen during implementation, but the field names must describe graph-domain semantics, not dataset artifacts.

Alternative considered: rename `GraphEvidenceNode` to `GraphDocumentSentenceNode` and keep the current fields. That would be honest for HotpotQA, but it would still force every future graph consumer to operate on a HotpotQA-shaped artifact, contradicting the cross-dataset goal.

### Decision: Projectors own dataset-to-request mapping, not domain packages

Dataset projectors may map HotpotQA `title`, `sentence_index`, and `position` into request fields such as `source_ref`, `group_key`, `sequence_index`, or `metadata`. Once that mapping is complete, graph builders, retrievers, models, evaluators, and validators operate on request/domain fields only.

Alternative considered: allow domain packages to call HotpotQA projectors as a convenience. The current implementation shows why this fails: batching, inference, and execution begin to reverse-build HotpotQA records to reuse lower-level APIs.

### Decision: Retrieval execution receives execution-ready tasks

`retrieval.execution.service` should not accept HotpotQA ranking records or choose dataset projectors. It should receive execution-ready task requests or a small domain task object containing the method request, task id, candidate ids, graph if needed, and token accounting data. Ranked-result assembly should use that domain task object or the relevant request, not a HotpotQA record.

Alternative considered: leave HotpotQA projection in retrieval execution because scripts currently load HotpotQA artifacts. That keeps dataset semantics inside the retrieval domain and makes adding a second dataset require editing retrieval internals.

### Decision: Training and model data use supervised ranking/domain tasks

Training-pair construction, Dense-FT data, R-GCN batching, R-GCN training, and R-GCN inference should consume dataset-neutral supervised ranking inputs: text requests, graphs, labels, and train-pair rows. Labels should be represented by `EvidenceLabel` or a dedicated supervised-task spec, not `HotpotQALabelRecord`.

Alternative considered: keep model APIs HotpotQA-specific and rely on future adapters later. That would make the current request-first contracts superficial because the trainable stack is one of the main consumers of graph/domain artifacts.

### Decision: Validators validate domain expectations, not HotpotQA records

Validation helpers for graphs, ranked results, train pairs, and importance artifacts should receive expected candidate IDs, expected labels, or request/domain specs. HotpotQA-specific validators remain in `validation.tasks` or dataset validation modules and are called before projection.

Alternative considered: keep validators tied to HotpotQA because current artifacts are HotpotQA. That makes validators unusable for any future dataset even when the runtime code has been made generic.

### Decision: Memory Stream importance is request/domain based

Memory Stream scoring already consumes `TemporalMemoryRankingRequest`, but importance content digests and validation still inspect HotpotQA candidate sentences. Importance helpers should use `TemporalMemoryRankingRequest` or a generic importance task spec with item ids, content text, source reference, and ordering metadata.

Alternative considered: treat importance artifacts as HotpotQA-only. That conflicts with Memory Stream being a retrieval method family rather than a dataset package.

### Decision: Stage/application code is the allowed dataset-aware boundary

`stages`, scripts, and application use cases may be HotpotQA-aware while the project only has HotpotQA artifacts. Their job is to load dataset artifacts, validate them with dataset validators, project them into domain requests/specs, and then call domain/model services.

Alternative considered: push dataset awareness into registries and method implementations. That spreads projection logic across unrelated packages and makes boundary tests weaker.

## Risks / Trade-offs

- Graph artifact schema changes can invalidate existing fixtures and cached graph files -> keep script file names stable and update tests/docs together; provide explicit migration notes for regenerated graph artifacts.
- Type churn across training/model code can be broad -> refactor in request-level batches and keep behavior-focused tests for pair records, dense-ft examples, R-GCN tensor batches, and inference rankings.
- Boundary tests can become too brittle -> scan only reusable production packages for dataset imports and dataset field names; allow dataset/stage/application files explicitly.
- Token accounting may drift when ranked-result assembly stops using HotpotQA records -> derive counts from `TextRankingRequest` and cover with focused tests.
- Memory Stream digest changes can invalidate existing importance artifacts -> document it as an intentional domain contract change and require stale-artifact failure messages.

## Migration Plan

1. Add failing boundary tests that identify HotpotQA imports and fields in reusable domain/model packages.
2. Generalize `MemoryGraph` node contracts and graph validation around dataset-neutral graph node fields.
3. Move request assembly out of retrieval execution and ranked-result assembly into dataset/application/stage or request-level helpers.
4. Convert training-pair, Dense-FT, and R-GCN APIs to supervised request/domain inputs.
5. Convert evaluation and failure-case APIs to `EvidenceEvaluationRequest` / `EvidenceLabel`.
6. Convert Memory Stream importance digest and validation to temporal request/domain inputs.
7. Update registry/stage/script payloads to perform HotpotQA projection at the boundary.
8. Update docs and run focused regression tests, full pytest, lint, type checking, and strict OpenSpec validation.

Rollback is batch-local: each consumer family should be migrated behind tests before moving to the next family, with public CLI behavior and method outputs checked after each batch.

## Open Questions

None for the proposal. The exact final graph node type name can be decided during implementation, but it must not encode HotpotQA/document-sentence semantics unless it is explicitly scoped to a dataset-specific adapter.

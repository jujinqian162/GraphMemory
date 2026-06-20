## Why

The request-first refactor moved HotpotQA records behind dataset projectors, but follow-up review found that HotpotQA sentence semantics still leak into graph, retrieval execution, training, model, evaluation, validation, and Memory Stream internals. This prevents future datasets from entering the domain/model layers through `GraphBuildRequest`, `TextRankingRequest`, `GraphRankingRequest`, `TemporalMemoryRankingRequest`, and `EvidenceEvaluationRequest` without satisfying HotpotQA-specific fields.

## What Changes

- Replace graph artifact node contracts that hard-code `document_sentence`, `source`, `sentence_id`, and `position` with dataset-neutral graph node semantics owned by the graph domain.
- Keep dataset-specific fields inside dataset records and projectors; after projection, domain/model/evaluation code consumes only consumer-owned request contracts or dataset-neutral task specs.
- Remove HotpotQA records and HotpotQA projectors from reusable domain/model packages, including retrieval execution, training pair construction, Dense-FT data, R-GCN batching/training/inference, evaluation/failure cases, validators, and Memory Stream importance handling.
- Move HotpotQA-aware projection and compatibility glue to dataset/application/stage boundaries where dataset ownership is explicit.
- Add boundary tests and documentation so future changes cannot reintroduce `HotpotQARankingRecord`, `HotpotQALabelRecord`, `candidate_sentences`, `gold_evidence_sentence_ids`, or document-sentence graph fields into reusable domain/model internals.
- Preserve current HotpotQA CLI flags, artifact file names, retrieval method names, ranking math, training math, metric formulas, and existing validated behavior except for intentional internal contract shape changes.

## Capabilities

### New Capabilities
- `dataset-neutral-domain-contracts`: Domain, model, retrieval, evaluation, and validation layers consume dataset-neutral contracts after dataset projection.
- `dataset-boundary-guardrails`: Architecture tests and durable docs enforce that dataset-specific records and fields remain at dataset/application/stage boundaries.

### Modified Capabilities

## Impact

- Affected production areas: `graph_memory/contracts/graphs.py`, `graph_memory/contracts/common.py`, `graph_memory/graphs/`, `graph_memory/retrieval/execution/`, `graph_memory/training_pairs/`, `graph_memory/models/dense_finetune/`, `graph_memory/models/graph_retriever/`, `graph_memory/evaluation/`, `graph_memory/validation/`, `graph_memory/retrieval/methods/memory_stream/`, `graph_memory/retrieval/tuning/`, registry payloads, stages, scripts, and tests.
- Dataset-specific HotpotQA record types remain valid inside `graph_memory/datasets/hotpotqa/`, scripts/stages that load HotpotQA artifacts, and explicit HotpotQA projectors.
- Public artifact compatibility should be preserved where practical through adapters or migration helpers; durable docs must clearly separate HotpotQA prepared artifacts from domain request/artifact contracts.
- No new runtime dependency is introduced.

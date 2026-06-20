## 1. Boundary Tests And Inventory

- [x] 1.1 Add architecture tests that fail when reusable domain/model packages import `graph_memory.datasets.hotpotqa.records` or `graph_memory.datasets.hotpotqa.projectors`.
- [x] 1.2 Add source guards for HotpotQA-specific required domain fields in reusable packages: `candidate_sentences`, `gold_evidence_sentence_ids`, `sentence_id`, `sentence_index`, `document_sentence`, `source`, and `position`.
- [x] 1.3 Add focused tests showing `GraphBuildRequest` can build a graph from non-HotpotQA item fields without requiring document-sentence fields.
- [x] 1.4 Add focused tests showing R-GCN inference ranks directly from `GraphRankingRequest` without constructing a `HotpotQARankingRecord`.
- [x] 1.5 Add focused tests showing evaluation and failure-case generation accept `EvidenceEvaluationRequest` / `EvidenceLabel` without HotpotQA label records.

## 2. Dataset-Neutral Graph Contracts

- [x] 2.1 Replace the generic `GraphEvidenceNode` shape with a dataset-neutral non-query graph node contract whose required fields are graph-domain fields.
- [x] 2.2 Update `NodeType` / node-kind validation so graph-domain contracts are not limited to `document_sentence`.
- [x] 2.3 Update `GraphBuilder` to map `GraphBuildNode` into the new graph node shape without hard-coding HotpotQA/document-sentence fields.
- [x] 2.4 Update graph construction context and edge rules to use abstract fields such as source reference, grouping key, sequence index, text, and metadata.
- [x] 2.5 Update graph views, statistics, tensorization, connectivity, tests, and docs to consume the new graph node shape.

## 3. Generic Validation Contracts

- [x] 3.1 Change graph validation to accept graph build requests or explicit expected graph item ids instead of HotpotQA ranking records.
- [x] 3.2 Change ranked-result validation to accept text ranking requests or explicit expected candidate ids instead of deriving ids from `candidate_sentences`.
- [x] 3.3 Change train-pair validation to accept evidence labels and expected candidate ids instead of HotpotQA label records.
- [x] 3.4 Keep HotpotQA record and label validators in the dataset-specific validation boundary with explicit HotpotQA names.
- [x] 3.5 Update all validation tests and error messages so dataset-neutral validators no longer mention HotpotQA.

## 4. Retrieval Execution And Result Assembly

- [x] 4.1 Move HotpotQA request projection out of `graph_memory.retrieval.execution.service` into stage/application or explicit dataset adapter code.
- [x] 4.2 Define an execution-ready ranking task/request wrapper if needed for method request, candidate ids, graph, latency, and token accounting.
- [x] 4.3 Update retrieval execution to accept built retrieval methods and request/domain inputs only.
- [x] 4.4 Update ranked-result assembly to calculate token counts from `TextRankingRequest` or domain inputs instead of HotpotQA records.
- [x] 4.5 Update graph-rerank and Memory Stream tuning helpers to consume projected request/domain inputs at the domain boundary.

## 5. Training Pairs And Dense-FT

- [x] 5.1 Introduce a dataset-neutral supervised ranking task or train-pair build request containing text request, evidence labels, graph, and candidate ids.
- [x] 5.2 Update train-pair builder and negative samplers to use the supervised request/domain input instead of HotpotQA records and projectors.
- [x] 5.3 Update Dense-FT example construction to consume `TextRankingRequest`, `EvidenceLabel`, and `TrainPairRecord` inputs.
- [x] 5.4 Update Dense-FT IR evaluator payload construction to consume text requests and evidence labels.
- [x] 5.5 Keep CLI/stage HotpotQA artifact loading stable by projecting HotpotQA records before calling training-pair or Dense-FT domain functions.

## 6. R-GCN Model Domain

- [x] 6.1 Update `TaskBatchInputs` and batching APIs to use text requests, graphs, labels, and pairs instead of HotpotQA ranking/label records.
- [x] 6.2 Remove the R-GCN inference reverse projection from `GraphRankingRequest` back into `HotpotQARankingRecord`.
- [x] 6.3 Update `train_graph_retriever` and dev evaluation to consume dataset-neutral training/dev task inputs.
- [x] 6.4 Update text embedding feature construction and seed-signal plumbing so batching can be driven directly by request/domain inputs.
- [x] 6.5 Preserve tensor ordering, relation vocab behavior, checkpoint schema, inference ranking, and training metrics through focused tests.

## 7. Evaluation And Failure Cases

- [x] 7.1 Remove the HotpotQA compatibility path from `evaluate_results`; callers must pass `EvidenceEvaluationRequest`.
- [x] 7.2 Update failure-case generation to consume `EvidenceEvaluationRequest` or `EvidenceLabel` instead of HotpotQA label records.
- [x] 7.3 Move HotpotQA label projection into evaluate stage/application code before evaluation service calls.
- [x] 7.4 Update evaluation tests to cover request-level inputs and current HotpotQA stage projection.

## 8. Memory Stream Importance Boundary

- [x] 8.1 Replace HotpotQA-based importance digest helpers with helpers based on `TemporalMemoryRankingRequest` or a generic importance task spec.
- [x] 8.2 Update importance artifact validation and selection to use temporal/domain expected ids and content digests.
- [x] 8.3 Update Memory Stream tuning and retrieval stage glue to project HotpotQA records before calling reusable Memory Stream helpers.
- [x] 8.4 Add stale-artifact tests for digest mismatch messages after the domain input change.

## 9. Registry, Stages, Scripts, And Docs

- [x] 9.1 Update registry payloads and stage functions so HotpotQA records are accepted only at dataset-aware boundaries and projected before reusable domain calls.
- [x] 9.2 Keep public CLI flags, file names, method names, config defaults, and artifact paths stable.
- [x] 9.3 Update contract, architecture, handoff, validation, and naming docs to describe dataset records, projectors, consumer requests, graph artifacts, supervised task specs, and forbidden dependency directions.
- [x] 9.4 Update historical docs only where current examples would mislead maintainers about active contracts.

## 10. Verification

- [x] 10.1 Run focused tests for graph contracts, validation, retrieval execution, training pairs, Dense-FT data, R-GCN batching/inference/training, evaluation, Memory Stream, and stage projection.
- [x] 10.2 Run architecture/source guards proving reusable domain/model packages do not import HotpotQA records/projectors or rely on HotpotQA field names.
- [x] 10.3 Run full test suite with `uv run pytest tests -q` outside the Windows filesystem sandbox.
- [x] 10.4 Run `uv run ruff check` and `uv run basedpyright --level error` outside the Windows filesystem sandbox.
- [x] 10.5 Run Python compile sweep, `git diff --check`, and `openspec validate --all --strict`.

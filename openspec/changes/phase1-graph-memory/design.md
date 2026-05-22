## Context

The repository currently contains documentation and a minimal Python scaffold, but no runnable Phase 1 evidence-tracing pipeline. The durable docs define the key constraints:

- Phase 1 is HotpotQA-only and retrieval-only.
- Model-visible artifacts must be separated from label artifacts.
- The package should use a library-core architecture with thin CLI adapters.
- Validation must fail fast instead of repairing or silently falling back.
- Dev tuning and test evaluation must remain separate.
- Commands and implementation handoff documentation are required after implementation.

The implementation will follow `docs/10-plans/phase1-real-graph-memory.md` and the contract/design/operations documents under `docs/`.

## Goals / Non-Goals

**Goals:**

- Implement a runnable Phase 1 pipeline from raw labeled HotpotQA data to metric CSVs.
- Preserve leakage-safe input/label separation across conversion, graph construction, retrieval, tuning, and evaluation.
- Build typed memory graphs using deterministic text/entity heuristics.
- Run BM25, frozen dense, BM25 graph rerank, and dense graph rerank under one ranked-result schema.
- Provide strict validation, reproducible configs, run summaries, command documentation, and implementation handoff documentation.
- Cover the core behavior with deterministic tests and skip dense model tests clearly when a local model is unavailable.

**Non-Goals:**

- Answer generation.
- Dense fine-tuning.
- Trainable GNN or graph retriever models.
- GraphRAG, Memory Stream, MemGPT-style memory, or Phase 2/3 baselines.
- 2WikiMultiHopQA, MuSiQue, or tool trajectory datasets.
- Persistent retrieval score caching.
- Deep plugin architecture or broad package hierarchy.

## Decisions

### Library Core With Thin CLI

Implement core behavior in `graph_memory/` and keep `scripts/*.py` as adapters that parse CLI arguments, read/write artifacts, validate boundaries, log progress, and write run summaries.

Rationale: this matches the existing architecture docs and makes converters, graph builders, retrievers, rerankers, tuning, and evaluators directly testable without invoking subprocesses.

Alternative considered: script-first implementation. It would be faster initially but would make validation, testing, and later baseline extensions harder to review.

### Flat Package With One Small Retrieval Subpackage

Use the package shape already documented:

```text
graph_memory/
  types.py
  validation.py
  io.py
  hotpotqa.py
  splits.py
  text.py
  entities.py
  graphs.py
  indexes/
    bm25.py
    dense.py
  retrieval.py
  rerank.py
  tuning.py
  evaluation.py
  observability.py
```

Rationale: Phase 1 benefits from clear files and stable boundaries without a plugin framework. `indexes/` is retained because BM25 and dense are parallel implementations.

### JSON Contracts At Boundaries, Dataclasses For Parsed Raw Data And Algorithm Values

Use `TypedDict` records for persisted JSON-like artifacts and frozen dataclasses for parsed raw dataset examples,
configs, ranked nodes, rerank results, and score components.

Rationale: artifact files remain transparent and validators own correctness, while algorithm signatures stay readable
and avoid deeply nested inline types. Raw JSON parsing should be dataset-specific, such as
`parse_hotpotqa_examples -> list[HotpotQAExample]`; Phase 1 should not introduce a generic JSON parser base class
that hides field-level semantics.

### Input And Label Artifacts Are Separate

`*_memory_tasks.input.json` is the only accepted task input for graph construction and retrieval. `*_memory_tasks.labels.json` is the only accepted gold source for evaluation and dev tuning. A combined `*_memory_tasks.json` compatibility output may be produced for humans, but retrieval and graph construction must not consume it.

Rationale: leakage prevention is primarily a data-contract boundary, not a complicated runtime access-control system.

### Graph Rerank Is A Scoring Transform

BM25 and dense retrievers produce complete initial rankings. Graph rerank consumes explicit initial scores plus a graph and config, then emits a complete reranked list.

Rationale: this avoids duplicating graph propagation logic for BM25 and dense methods and leaves a clear future boundary for score caching if runtime later becomes a real blocker.

### Evaluation Uses Shared Graphs

Connected evidence and query-evidence connectivity are computed from the constructed graph and each method's selected top-k node IDs, not from method-emitted edges alone.

Rationale: flat methods should be evaluated fairly on the same graph structure even when their own outputs do not contain graph edges.

### Run Summaries Are Script-Level Obligations

Every script writes a compact run summary with effective config, input/output paths, counts, timings, environment notes, and status.

Rationale: reproducibility should be traceable from artifacts and command docs without adding a monitoring framework.

## Risks / Trade-offs

- Dense retrieval may require a local Sentence-Transformers model. -> Tests use a fake encoder or skip real-model checks clearly when the model is unavailable; commands document the encoder setting.
- Dev grid search may be slow without score caching. -> Phase 1 prioritizes correctness and inspectability; caching is deferred until the pipeline is stable.
- Heuristic entity extraction may be weaker than full NER. -> spaCy remains optional, deterministic heuristics are tested, and graph construction is explicit about the chosen config.
- The original project plan uses combined memory-task artifacts. -> The implementation writes optional compatibility artifacts while the runnable pipeline documents and enforces leakage-safe `.input.json` and `.labels.json` paths.
- OpenSpec specs may be broader than unit tests. -> Tasks map requirements to test files and command docs so review can trace each behavior to code and verification.

## Migration Plan

1. Create the package, scripts, configs, and tests from the existing scaffold.
2. Implement Phase 1 task steps in test-first order.
3. Update command and handoff documentation with real paths, functions, and verification output.
4. Validate OpenSpec and run the test suite before committing.

Rollback is simple before generated experiment artifacts are shared: revert the feature branch or individual commits. No external service or database migration is involved.

## Open Questions

None for initial implementation. If the dense model is not available locally during verification, report the skip/blocker explicitly instead of changing the scientific default silently.

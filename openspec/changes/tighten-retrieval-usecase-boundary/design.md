## Context

The package refactor has already created domain packages, typed retrieval request families, and final architecture tests. The remaining retrieval gap is at the highest run path: `graph_memory.retrieval.execution.service.run_retrieval()` still accepts CLI-shaped parameters such as `encoder_model`, `query_prefix`, `passage_prefix`, `checkpoint_path`, and `device`; it also resolves method requests and builds retrieval method objects.

That puts application use-case orchestration inside the retrieval execution package. It keeps the old `RetrievalBuildContext` pressure alive even though the explicit context class has been removed. The original core refactor design requires a Single Level of Abstraction: high-level retrieval run orchestration should resolve a method and execute ranking, not carry dense prefix strings or checkpoint runtime details as loose parameters.

The external compatibility boundary is unchanged. Scripts keep their CLI flags and artifact behavior. Internal imports and service signatures may change.

## Goals / Non-Goals

**Goals:**

- Introduce `graph_memory.application.run_retrieval` as the use-case orchestration layer.
- Represent a retrieval run with one `RunRetrievalRequest` object that contains typed runtime/config objects instead of loose dense prefix parameters.
- Narrow `graph_memory.retrieval.execution.service` to execute an already-built `RetrievalMethod` and assemble ranked artifacts.
- Keep method-family request resolution and object construction before execution.
- Move graph-rerank initial score precomputation out of the execution service and make tuning accept typed dense runtime state.
- Update architecture tests so they fail if retrieval execution receives loose dense/checkpoint parameters again or if domain packages import root workflow integration ports.
- Update durable docs so the application boundary is no longer silently missing.

**Non-Goals:**

- Do not change CLI argument names, defaults, choices, or required flags.
- Do not change public retrieval method names, ranking formulas, top-k semantics, graph-rerank tuning objective, artifact schemas, or checkpoint schema.
- Do not introduce a generic pipeline engine, dependency injection container, plugin discovery, or broad compatibility facade.
- Do not restructure unrelated dataset, graph, evaluation, training-pair, or model internals beyond fixing direct root-port imports discovered in the review.

## Decisions

### Decision: Application owns complete retrieval run orchestration

`graph_memory.application.run_retrieval` will define `RunRetrievalRequest` and `run_retrieval(request)`. Scripts construct the request after CLI parsing and file loading. The application service validates the use-case request, resolves a precise `MethodBuildRequest`, builds the `RetrievalMethod`, and calls retrieval execution.

Alternative considered: keep `run_retrieval` in `retrieval.execution.service` and only rename parameters. That keeps the same mixed layer and does not satisfy Single Level of Abstraction.

### Decision: Retrieval execution receives a built method

`retrieval.execution.service.run_retrieval` will accept `retrieval_method`, `method`, `task_inputs`, and `top_k`. It may validate task inputs, measure per-task latency, call `rank_task`, assemble artifacts, and validate ranked results. It will not know about dense prefixes, checkpoint paths, graph configs, resolver requests, or method factory construction.

Alternative considered: pass `MethodBuildRequest` into execution and let execution call the factory. That is better than loose parameters, but still combines object construction and execution.

### Decision: Keep typed resolver input at the resolution boundary

`RetrievalMethodResolveRequest` can remain a typed request object used by application-to-retrieval resolution. Its dense settings stay inside `DenseRuntime`; graph inputs use `GraphIndex` after validation; trainable runtime details are grouped into `TrainableGraphRuntime`. The important boundary is that resolver output is precise and execution/factory do not receive a universal optional-field bag.

Alternative considered: move resolver into `application`. That would make retrieval domain smaller, but it would also split registry-driven method-family knowledge away from retrieval ownership. Keeping resolver in retrieval is acceptable as long as execution stays lower-level.

### Decision: Tuning precomputes initial scores through a tuning-owned helper

Graph-rerank tuning repeatedly needs seed scores, but initial-score precomputation is not per-task execution of a final method. Move the cache type and helper into `retrieval.tuning.initial_scores`, and pass `DenseRuntime` rather than loose `encoder_model`, `query_prefix`, and `passage_prefix` through tuning internals.

Alternative considered: leave the helper in execution because it loops over tasks. That makes execution a dumping ground for adjacent workflows and keeps dense loose parameters in the wrong layer.

### Decision: Root workflow integration ports are one-way adapters only

Domain packages should import owned implementation modules directly, not the retained root ports `graph_memory.io` and `graph_memory.observability`. The existing architecture test will be strengthened to catch domain imports from root workflow integration ports.

Alternative considered: allow root ports as a stable public library facade. That contradicts the refactor design: those files were retained only for workflow integration, not as general internal dependencies.

## Risks / Trade-offs

- Internal tests importing `graph_memory.retrieval.run_retrieval` will need to move to the application use case -> update tests to reflect the intended boundary instead of preserving a misleading re-export.
- `RunRetrievalRequest` is still a broad use-case request -> keep broadness at application only, and prevent loose dense fields from entering retrieval execution/factory.
- Tuning script still exposes dense CLI flags -> convert them to `DenseRuntime` at the script/application edge while preserving parser contracts.
- Architecture tests can become brittle if they check every field name everywhere -> scope them to production boundary files and semantic imports rather than freezing unrelated config record fields.

## Migration Plan

1. Add failing tests for the new application run request, narrowed execution signature, tuning dense runtime boundary, and forbidden domain imports from root ports.
2. Add `graph_memory/application/` and move complete retrieval run orchestration into `application/run_retrieval.py`.
3. Narrow `retrieval/execution/service.py` and update exports.
4. Move initial-score precomputation into `retrieval/tuning/initial_scores.py` and update tuning service/scripts.
5. Update scripts and tests to import the application use case.
6. Fix direct domain imports of root integration ports.
7. Update durable docs and run focused tests, lint/type validation where available, and OpenSpec strict validation.

## Open Questions

None. The user has approved the direction: make retrieval parameters clean according to the previous review and fix the discovered documentation and guardrail gaps.

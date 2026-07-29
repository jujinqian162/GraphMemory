## 1. Canonical trajectory domain (M1)

- [x] 1.1 Add `graph_memory/trajectories/` closed Pydantic contracts for source intents, tool definitions, source spans, ordered message/tool-call/tool-output events, and canonical trajectories.
- [x] 1.2 Enforce stable IDs, strict event ordering, unique call IDs, exact call/output pairing, matching tool names, and source-reported-success semantics in the owning models.
- [x] 1.3 Add canonical serialization/fingerprint helpers that are deterministic and contain no dataset-specific raw objects.

## 2. ISETrace adapter (M1)

- [x] 2.1 Add strict dataset-owned ISETrace raw record/message/tool models and streaming JSONL parsing.
- [x] 2.2 Add deterministic conversion from embedded intents, tool definitions, and OpenAI-format messages into `CanonicalTrajectory`.
- [x] 2.3 Add compact ingestion counters for accepted/rejected records, rejection reasons, calls, outputs, argument parse failures, and source-success/error-text conflicts; do not add a separate audit framework or CLI.
- [x] 2.4 Add sample-derived tests covering multiple intents, multiple calls in one assistant message, malformed arguments, orphan/duplicate outputs, stable IDs, and preservation of message spans.

## 3. Query-independent provenance graph (M2)

- [x] 3.1 Add extensible namespaced graph node/edge contracts with source spans, derivation metadata, core endpoint validation, stable graph fingerprints, and no edge weights.
- [x] 3.2 Add a core graph builder for `execution.tool_call`, `execution.tool_output`, `resource.artifact`, `execution.returns`, and `temporal.precedes`.
- [x] 3.3 Add plugin-oriented explicit artifact extraction for read/write/edit/web-fetch path or URL arguments and emit `resource.reads`/`resource.writes`.
- [x] 3.4 Add conservative exact binding extraction and unique-producer `data.feeds` edges for structured leaves and typed high-information output tokens.
- [x] 3.5 Add tests proving graph IDs/fingerprints are query-independent, core endpoints are valid, future namespaced semantic kinds with source spans are representable, and unsupported core-like relations fail closed.

## 4. Provenance motifs and template queries (M2)

- [x] 4.1 Add separate `MotifSpec`, logical dependency, query intent, template metadata, and synthetic query contracts; labels/hidden slots must not enter graph or rendered query inputs.
- [x] 4.2 Add deterministic extractors for call-result, value-flow, artifact-lifecycle, multi-hop-flow, and multi-source-join motifs where graph structure supports them.
- [x] 4.3 Add a versioned template catalog with at least six unique templates and three style tags per supported motif/query-intent pair.
- [x] 4.4 Add deterministic query verbalization keyed by motif ID/template ID/seed, strict safe-slot rendering, and duplicate-template validation.
- [x] 4.5 Add tests for motif support minimality, multiple query intents over one motif, wording diversity, answer/node-ID leakage rejection, and graph fingerprint invariance across generated queries.

## 5. Documentation and verification

- [x] 5.1 Update maintained architecture/data-contract docs with canonical trajectory ownership, query-independent graph vocabulary, namespaced extension semantics, and the deliberate no-workflow/no-model boundary.
- [x] 5.2 Run focused tests for ISETrace/trajectory/provenance/query synthesis.
- [x] 5.3 Run `uv run pytest -q`, `uv run ruff check .`, the repository-standard `uv run basedpyright --level error`, and strict focused BasedPyright over all new paths.
- [x] 5.4 Mark this task list complete and stop before experiment integration, model training, LLM query generation, or paper-result work.

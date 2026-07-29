## Context

ISETrace stores one completed session per JSONL record. A record contains one or more source intents, the available tool definitions, OpenAI-style messages, assistant tool calls with JSON-string arguments, tool outputs linked by `tool_call_id`, and the final response. The downloaded sample has complete call/output pairing and parseable arguments, but outputs are unstructured strings and the source `success` field is not an authoritative execution outcome: successful flags may accompany shell tracebacks or structured error text.

The repository currently has evidence-specific graph contracts only. The deleted execution-provenance schema mixed a query Task node, answer/claim types, semantic edge weights, and gold-conditioned graph construction. None of those contracts will be restored.

## Goals / Non-Goals

**Goals:**

- Preserve enough ordered source content for future semantic annotation while exposing a clean canonical tool-execution view now.
- Build one stable graph per trajectory, reusable by any number of later queries.
- Provide a small core graph vocabulary while allowing later namespaced node and relation kinds.
- Derive only native or deterministic high-confidence relations; never derive graph topology from a query or support label.
- Extract several motif shapes and several semantic query intents per shape.
- Make query wording extensible and diverse without permitting templates to access hidden answer values or node IDs.

**Non-Goals:**

- No R-GCN, Dense-FT, EPGM, retrieval method, metric, or experiment workflow integration.
- No LLM-generated validation/test queries.
- No claim/decision classifier in this change; only source-span and namespaced extension support.
- No semantic-similarity edges, shell AST parser, or authoritative failure/retry labels.
- No full data-quality platform. Ingestion exposes compact counters and structured rejection reasons only.
- No General-AgentBench or Toolathlon adapter.

## Decisions

### D1: Canonical trajectory is an ordered event stream

`CanonicalTrajectory` contains source identity, source intents, tool definitions, ordered canonical events, final response, and source metadata. Events are a discriminated union:

- `MessageEvent`: system, user, or assistant text plus optional assistant reasoning text;
- `ToolCallEvent`: call ID, tool name, parsed JSON arguments, raw arguments, and source position;
- `ToolOutputEvent`: call ID, tool name, output text, and `source_reported_success`.

Every event has a stable `event_id`, `message_index`, and `sub_index`. Tool calls embedded in an assistant message are flattened after that message event in source order. The trajectory validator enforces unique IDs, total ordering, unique call IDs, exactly one output per call, output-after-call ordering, and matching tool names.

The field is named `source_reported_success`, not `success`, because ISETrace's flag is preserved source metadata rather than a derived semantic judgment.

Alternative considered: store only paired `ToolExecution` rows. Rejected because future claim/decision extraction needs stable message text spans, and flattening away assistant/user text would force a second raw-message representation later.

### D2: Raw ISETrace stays dataset-owned; canonical contracts are dataset-neutral

`graph_memory/datasets/isetrace/` owns strict source records, JSONL iteration, conversion, and compact ingestion counters. `graph_memory/trajectories/` owns canonical records and source-span types so a later General-AgentBench adapter can project into the same domain.

The adapter uses each trajectory's embedded `source_intents` as authoritative for normalization. The separate intents corpus remains available for later corpus-level consistency checks but is not required to adapt a trajectory.

### D3: Graph kinds are namespaced strings with a validated minimal core

The graph stores generic closed node/edge records whose `kind`/`relation` fields are validated namespaced identifiers rather than a closed enum. Version 1 emits only:

**Nodes**

- `execution.tool_call`
- `execution.tool_output`
- `resource.artifact`

**Edges**

- `execution.returns`: call to its output;
- `temporal.precedes`: one call to the next call;
- `data.feeds`: output to a later call when a unique high-information value is reused;
- `resource.reads`: call to an explicitly referenced artifact;
- `resource.writes`: call to an explicitly written artifact.

Core endpoint invariants are validated. Unknown namespaced kinds remain representable for later annotation layers, e.g. `semantic.claim`, `semantic.decision`, `semantic.supports`, but the core builder never emits them.

Alternative considered: a discriminated union closed over all node types. Rejected because every later semantic annotation type would require changing the base graph serialization union and all consumers.

### D4: Extension nodes retain source-span provenance

Every graph node can carry one or more `SourceSpan` records containing `event_id`, optional character offsets, and an optional JSON pointer. Core execution nodes reference their canonical events. Future NLP annotators can add semantic nodes that point to exact assistant/user/tool-output text spans instead of mutating the source event or pretending that claims were native trace fields.

Node and edge `attributes` are JSON-valued metadata. They are not interpreted as model features by this change. Edges also record `derivation` (`native`, `deterministic`, or future `annotator`) and `extractor` identity. There is no edge weight.

### D5: Artifact extraction is explicit and plugin-oriented

The v1 builder recognizes explicit resource arguments for the sampled ISETrace tools:

- `read`: reads `path`;
- `write`: writes `path`;
- `edit`: reads and writes `path`;
- `web_fetch`: reads `url`.

Artifact IDs are trajectory-local hashes of `(artifact_kind, canonical_value)`. Tool-specific extractors implement a small protocol and can be registered later; v1 does not parse arbitrary shell commands.

### D6: Data-flow extraction uses exact, high-information bindings only

The builder extracts candidate bindings from structured JSON output leaves when an output is JSON, and from conservative typed text tokens such as absolute paths, URLs, UUIDs, long hashes, and filenames otherwise. A binding creates `data.feeds` only when:

- the producer output precedes the consumer call;
- the value occurs in a downstream argument leaf or canonical argument text;
- the value passes minimum length/information filters;
- the producer is unique within the trajectory for that value.

The edge records only a hash/type of the binding, not a query relevance score. Common booleans, nulls, short numbers, and generic words are excluded. Semantic similarity never creates gold topology.

### D7: Graph identity excludes queries and labels

`ProvenanceGraph.graph_id` equals the canonical `trajectory_id` and records a digest of the canonical trajectory. Graph serialization contains no query ID, query text, answer, support ID, motif type, or template metadata. Multiple query examples reference the same graph ID.

A graph fingerprint is computed from graph schema version, trajectory digest, nodes, and edges. Tests construct different queries over one graph and require the fingerprint to remain unchanged.

### D8: Motif specs and query text are separate artifacts

`MotifSpec` contains:

- motif ID/type and graph ID;
- ordered anchor/target roles;
- answer output IDs;
- complete support output IDs;
- logical support dependencies;
- safe public slots;
- hidden binding metadata.

The graph never contains a motif spec. The verbalizer receives only `safe_slots` plus the selected template; it cannot access node IDs, hidden values, or labels.

V1 extracts these motif families where their required edges exist:

1. `call_result`: a call and its returned output;
2. `value_flow`: producer output feeds a downstream call and its result;
3. `artifact_lifecycle`: writer and later reader joined through an artifact;
4. `multi_hop_flow`: two or more composable value/artifact dependencies;
5. `multi_source_join`: two or more producer outputs feed one downstream call.

`call_result` is useful for ingestion/query-synthesis testing but must be reported separately from dependency motifs in future experiments. Failure/recovery is deliberately excluded until an outcome classifier exists.

### D9: Query diversity is represented by query intent, not only motif shape

A motif can support multiple `QueryIntent` values, for example:

- identify an upstream producer;
- identify the downstream consumer result;
- recover the complete dependency chain;
- identify an artifact origin;
- identify an artifact's later use;
- explain which sources were combined.

The versioned template catalog contains multiple syntactic forms and style tags per `(motif_type, query_intent)`: direct question, retrospective memory request, concise imperative, passive voice, and provenance/explanation wording. Deterministic selection is keyed by motif ID and generation seed.

The initial catalog must provide at least six templates for every supported `(motif_type, query_intent)` pair and at least three style tags. Tests enforce normalized-text uniqueness and require one motif to produce multiple distinct query forms across template IDs.

Alternative considered: concatenate synonym lists combinatorially. Rejected because it produces a large number of grammatically weak near-duplicates and makes template count a misleading diversity measure.

### D10: M1/M2 stop at domain artifacts

This change exposes library APIs to:

```text
raw ISETrace JSONL
  -> CanonicalTrajectory
  -> ProvenanceGraph
  -> MotifSpec
  -> SyntheticProvenanceQuery
```

It does not add ISETrace to `DatasetName`, `DatasetId`, Hydra configs, Prefect stages, retrieval requests, training pairs, or evaluation requests. Those joins require a later design for fixed splits, candidate text, model inputs, and natural-query validation.

## Risks / Trade-offs

- **Open namespaced kinds weaken compile-time exhaustiveness.** Core constants and graph validators enforce the v1 vocabulary produced by the core builder; consumers must explicitly declare supported kinds rather than silently accepting unknown semantic extensions.
- **Exact feeds may have lower recall.** Precision is preferred because these edges become pseudo labels. Coverage is surfaced in compact ingestion/build summaries and can be expanded through new deterministic extractors later.
- **Templates can still induce shortcuts.** Query intent, style tags, template IDs, and motif family are retained for future split/ablation analysis; a later LLM/human test set remains required.
- **Artifact semantics are tool-specific.** V1 handles only explicit path/URL arguments. It intentionally avoids claiming that shell-string parsing is reliable.
- **Canonical event records duplicate some source text.** This is accepted to retain stable future annotation spans; raw source locators and compact normalized fields avoid preserving arbitrary untyped message objects.

## Migration Plan

1. Commit the completed deletion of the legacy RQ2 stack as an independent baseline.
2. Add canonical trajectory and ISETrace adapter contracts with sample-derived tests.
3. Add graph contracts, builder, artifact/data-flow extractors, and query-independence tests.
4. Add motif contracts/extractors and the versioned query template catalog.
5. Update maintained architecture/contracts documentation and run focused plus full verification.

Rollback removes only the new packages/docs/tests. The evidence workflow is not modified.

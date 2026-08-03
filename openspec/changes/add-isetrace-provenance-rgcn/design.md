## Context

The current branch already provides the reusable parts needed by a trainable RQ2 method:

- `CanonicalTrajectory` and the ISETrace adapter;
- one query-independent `ProvenanceGraph` per trajectory;
- content-level argument/output candidates with exact source spans;
- deterministic motif extraction with separate focus and participant output IDs;
- natural v7 query records compiled to exact source spans;
- frozen E5 encoding, relation-aware graph convolution, disconnected-union graph batching, pair sampling, training, checkpointing, and dev selection for RQ1 evidence R-GCN;
- training-free `provenance_path` retrieval and exact-span ISETrace evaluation.

The old execution-provenance R-GCN was intentionally deleted. It expected a query Task node inside a label-conditioned graph, ranked only ToolOutput containers, and jointly predicted graph edges. Those contracts do not match the current ISETrace graph or v7 span labels and will not be restored.

The natural-query authoring file is now a general corpus rather than a preassigned test file. The first trainable implementation therefore needs two additions: a simple deterministic natural/template data composition boundary and a thin provenance adapter around the existing R-GCN runtime.

## Goals / Non-Goals

**Goals:**

- Keep the user-facing ISETrace query configuration small and use exact per-origin task counts.
- Split natural queries deterministically without placing two queries over the same trajectory in different splits.
- Inject natural queries into train and dev while keeping test natural-only.
- Generate template supervision from existing motifs without changing graph construction.
- Reuse the existing RQ1 R-GCN neural, batching, sampling, training, checkpoint, and evaluation components.
- Rank the same provenance content units used by `provenance_path` and preserve exact source-span evaluation.
- Keep natural and template origins visible in artifacts and metrics but unavailable as model features.

**Non-Goals:**

- No configurable split strategy, grouping policy, invalid-record policy, manifest name, schema version, or extension policy.
- No query-dependent persisted graph, EvidenceGraph compatibility projection, or second R-GCN implementation.
- No Dense-FT or jointly fine-tuned text encoder in the first version.
- No dependency-edge prediction head, edge discovery claim, path/edge metric, or new graph annotation.
- No semantic parser for arbitrary ToolOutput text and no LLM use in template generation.
- No PyG, DGL, Accelerate, Lightning, distributed trainer, or generic data-mixture framework.
- No formal paper result until natural queries are sufficiently reviewed and the configured multi-seed runs complete.

## Decisions

### 1. Expose sources and exact query counts in ISETrace configuration

The ISETrace dataset configuration uses this scientific surface:

```yaml
name: isetrace
trajectory_source: data/isetrace/raw/trajectories
natural_query_source: data/isetrace/query-authoring/isetrace-v7-raw.jsonl

queries:
  splits:
    train: {natural: 2692, template: 5384}
    dev: {natural: 393, template: 393}
    test: {natural: 981, template: 0}

chunking:
  tokenizer_name: models/intfloat-e5-base-v2
  max_tokens: 512
  reserved_tokens: 8
  overlap_tokens: 64
```

Counts are strict nonnegative integers and mean exactly what preparation emits. Train/dev may set `natural: 0` for template-only supervision. Test requires natural greater than zero and template equal to zero. Generic evidence-dataset profile caps do not alter ISETrace counts.

The following are fixed implementation behavior and do not appear in config:

- v7 natural-query parsing;
- authoring metadata sidecar discovery;
- invalid-query exclusion and reporting;
- grouping by trajectory;
- split artifact identity;
- test natural-only behavior;
- template generator identity;
- corpus-extension handling;
- schema and implementation versions.

The pinned source revision is owned by repository dataset registration. When an authoring run sidecar is available, preparation verifies that its recorded revision and source digests agree with the registered/raw source. A mismatch fails before graph construction. `source_revision` is removed from the user-facing ISETrace dataset config.

This is an explicit replacement of the earlier `isetrace-nontrain-benchmark` dataset contract. The old test-only `splits.<name>.kind/source/offset` shape, config-level `allow_unreviewed`/`accepted_only` admission policy, and `answer_only`/`support`/`intent_aware` target policy are retired rather than adapted. Natural v7 exact spans become the authoritative natural labels; template focus records remain training-only labels. Review state is managed by the natural corpus release process: engineering runs may use the generated corpus, while a formal run must pin a separately reviewed natural source and its digest. No compatibility parser accepts the retired fields.

Alternative: keep the old generic split and policy shape and point all splits at one file. Rejected because it would expose fixed mechanics, encourage row-level slicing, retain output-level label semantics that no longer match exact spans, and cannot safely prevent same-trajectory leakage.

### 2. Resolve natural queries before registered trajectory-grouped ownership

Preparation first parses the natural corpus and resolves every usable query to exactly one canonical trajectory and exact source spans. Invalid, unmatched, uncompilable, or ambiguous records are excluded deterministically and counted.

All natural queries mapped to one `trajectory_id` form one indivisible group. Groups are assigned deterministically from `split_seed` toward exact targets derived from repository-owned weights. For the 4,066-query server corpus these resolve to 2,692 train, 393 dev, and 981 test records. A changed source digest receives a newly content-addressed deterministic plan. The training seed and requested supervision counts never change ownership. The prepared artifact records source identity, registered weights, resolved targets, requested/available/selected origin counts, actual trajectory counts, and dropped-record counts.

Grouping is fixed at trajectory granularity because the model receives the complete graph. Query-row splitting would leak an identical graph across train and test. A stricter unseen-intent study, if later required, is a separate benchmark change rather than a runtime switch.

### 3. Select exact natural/template counts

For each split, preparation deterministically selects exactly the configured natural count and exactly the configured template count from records whose trajectories belong to that split. Natural and template may independently be zero for train/dev, but each split must contain at least one task. Template-only preparation still builds deterministic graph anchors from its frozen trajectory partition.

If either requested count exceeds its eligible pool, preparation fails with requested and available counts rather than silently truncating. Content-addressed identity includes exact query counts and registered ownership weights. Retired `split_ratio` and `mix_ratio` fields are rejected explicitly.

A size-matched template-only control against the 1:2 experiment uses 8,076 train templates and 786 dev templates, with no natural train/dev tasks and the same 981-query natural test.

### 4. Generate simple template supervision from existing motifs

The template generator consumes existing `MotifSpec` and `MotifAuthoringTarget` values. It does not add a second motif extractor. Dependency motifs (`value_flow`, `artifact_lifecycle`, `multi_hop_flow`, and `multi_source_join`) are eligible; `call_result` remains outside the primary dependency-training pool.

A small internal renderer uses safe descriptions derived from the existing graph, such as tool names, artifact names, dependency direction, and hop count. Rendered text cannot contain graph IDs, event IDs, node IDs, binding hashes, or hidden answer values. Template implementation identity participates in artifact cache identity but is not a Hydra field.

For each target:

- `focus_output_ids` define positive ToolOutput owners;
- output-content children reached through `execution.has_content` are positive retrieval candidates;
- `participant_output_ids` provide structural context only and are not positive unless also focused;
- all other provenance content candidates remain available for hard-negative sampling.

This deliberately reuses the current focus/participant distinction and avoids restoring legacy answer/support contracts. Natural train/dev queries continue to use exact v7 spans, including argument evidence where annotated. Template labels provide coarse but deterministic output-content supervision; natural injection supplies fact-level and argument-level supervision.

Template query and label records carry `query_origin=template`; compiled v7 records carry `query_origin=natural`. Origin, motif, and template metadata are available only for preparation summaries and stratified metrics.

### 5. Reuse the existing RQ1 graph-retriever runtime

The new public method identity is `provenance_rgcn`. It supports only the execution-provenance task family and is not an alias for the deleted provenance method.

The implementation reuses:

- the frozen dense encoder interface and embedding provider;
- `TaskGraphTensor` and disconnected-union `GraphBatch`;
- `RGCNGraphEncoder`, typed/shared relation transforms, and the existing node scorer;
- map-style task datasets, seeded DataLoaders, device transfer, optimizer loop, gradient clipping, metric records, and checkpoint helpers;
- existing easy-random, BM25-hard, Dense-hard, and graph-neighbor negative-sampling behavior;
- existing R-GCN selection metric machinery and exact-span evaluation.

A provenance tensorizer is required because `ProvenanceGraph` is a distinct domain model and must not be converted into `EvidenceGraph`. The adapter emits the same low-level graph tensors expected by the reused neural/runtime components.

Alternative: restore the deleted `models/provenance_rgcn` stack. Rejected because it duplicates the maintained RQ1 runtime and restores stale query-node, output-only, and edge-head assumptions.

### 6. Add an ephemeral disconnected query node only at tensorization

The persisted provenance graph remains byte-identical for every query over a trajectory. For one ranking task, tensorization appends one ephemeral query node containing the prefixed query embedding. It is disconnected from provenance edges and exists only to satisfy the existing task-graph/scorer contract:

```text
provenance graph nodes + one ephemeral query node
    -> R-GCN over provenance edges
    -> candidate/query gather
    -> existing node scorer
```

No query-overlap edge or query-derived topology is created. The query node receives only its self transformation; candidate conditioning happens in the scorer through `[h_candidate, h_query, h_candidate * h_query, features]`.

This preserves graph query independence while avoiding a new query-vector batching abstraction.

### 7. Tensorize the physical provenance relations with one fixed policy

The first version uses the existing graph edges directly. Each enabled physical relation becomes explicit forward and reverse message relations with uniform weight `1.0`:

- `execution.returns`;
- `execution.has_argument`;
- `execution.has_content`;
- `data.feeds`;
- `resource.reads`;
- `resource.writes`;
- `content.next` when chunking emits it.

`temporal.precedes` is excluded, matching `provenance_path`, because it connects nearly every call and can turn chronological position into a shortcut. The policy is fixed in code; there is no enabled-relation list or edge-weight policy in config.

Node embeddings use the frozen text encoder. The first version adds no node-kind embedding or position feature; relation types provide the structural signal. Node IDs, message ordinals, motif types, template identities, query origins, and split metadata never become model features.

The initial method uses the existing configurable R-GCN layer count. No logical shortcut edges or new persisted relations are added in this change.

### 8. Reuse pairwise node supervision and keep the model node-ranking only

Natural labels are mapped to candidates by exact source-span overlap. Template labels use focused output-content candidates. The existing train-pair builder semantics are reused through a provenance task adapter:

- every mapped positive is materialized;
- easy random, BM25, Dense, and graph-neighbor negatives use the configured existing counts;
- graph-neighbor negatives may include motif participants that are not focused, teaching that graph participation alone does not imply relevance.

The existing binary node-ranking loss and checkpoint format remain authoritative unless current strict validation requires adding the new method identity or provenance relation vocabulary. No edge scorer or auxiliary dependency loss is introduced. Retrieved native edges/path information remains a diagnostic projection from the input graph, as for the training-free method.

### 9. Select by available dev origin and evaluate natural-only test

Training consumes the configured exact train composition. Mixed dev containing natural records selects checkpoints on natural Recall@5 and reports template metrics separately. Template-only dev selects on template Recall@5 and records template as the selection origin; this enables the explicit synthetic-supervision ablation without inventing empty natural metrics.

Formal test contains only natural queries. It uses the ISETrace exact-span suite: Recall/Coverage, Coverage@512/1024/2048 Tokens, Full Support, span F1, MRR, and evidence density. Path and edge metrics remain unavailable without independent labels.

Predictions and metrics are additionally grouped by `query_origin` for train/dev diagnostics and by existing natural-query memory mode when metadata is available. Origin never enters retrieval requests or model tensors.

### 10. Integrate one trainable provenance method into the existing workflow

`provenance_rgcn` requires train, dev, and test prepared splits. The workflow follows the existing trainable lifecycle:

```text
prepare mixed splits
  -> materialize provenance candidates/graphs
  -> build train pairs
  -> encode frozen node/query text
  -> train/select checkpoint
  -> retrieve natural test
  -> exact-span evaluate
```

Registry gains one method definition, settings type, payload, and builder for the execution-provenance family. The retrieval workflow matrix is explicitly extended so `provenance_rgcn` may consume the query-independent physical provenance graph and execute train/select/retrieve, while BM25, Dense, and GraphRAG still receive no provenance graph and `provenance_path` remains training-free. Those non-training methods evaluate only the derived natural test split and schedule no pair-building or training stage. Evidence R-GCN and Dense-FT behavior is unchanged.

No compatibility loader is added for the deleted provenance checkpoints. The new checkpoint is produced and consumed only by the current workflow.

## Risks / Trade-offs

- **Template wording may still create shortcuts.** Natural queries are injected into train/dev, checkpoint selection is natural-dev-first, participant-but-nonfocus negatives are retained, and formal test is natural-only.
- **Template labels can cover long outputs.** The first implementation accepts coarse focused-output supervision to remain simple; natural exact-span training examples provide finer supervision. Fact-level deterministic template parsing is deferred unless results show it is necessary.
- **Physical provenance paths are longer than RQ1 evidence edges.** The layer count remains an ordinary model parameter, while no shortcut relation or second graph is added before a measured need exists.
- **Tool distribution is skewed.** Existing hard negatives and trajectory grouping reduce trivial memorization; tool-balancing policy is deferred until observed train/dev diagnostics justify it.
- **Natural corpus growth changes resolved split identity.** Source content identity makes the new deterministic target plan explicit. Formal runs must record the new source and split summary.
- **Existing unreviewed natural records are not formal gold.** Engineering runs may use them, but paper claims require the selected natural test corpus to be reviewed and frozen.

## Migration Plan

1. Add failing config and preparation tests for explicit ISETrace origin counts, source revision inference, registered trajectory-grouped ownership, exact counts, template-only train/dev, and natural-only test.
2. Add training-only template query records and a small renderer over existing motifs, then materialize focused output-content labels without changing `ProvenanceGraph`.
3. Add provenance tensorization into the existing graph-batch contract, including the ephemeral disconnected query node, fixed relation mapping, candidate ownership, and source-span label mapping.
4. Add `provenance_rgcn` model config, Registry definition/builder, pair/training adapters, strict checkpoint round trip, and batch-size-one inference.
5. Wire train/dev/test workflow stages, origin-aware dev checkpoint selection, natural-only test retrieval, Coverage@512/1024/2048, and origin-stratified summaries.
6. Add smoke/quick/full configuration, maintained docs, and focused architecture/config/workflow/model/evaluation tests.
7. Run Ruff, BasedPyright, focused pytest, broad pytest, strict OpenSpec validation when the CLI is available, and `git diff --check` before any formal training run.

## Open Questions

None blocking. The implementation intentionally uses one registered split ownership policy, exact per-origin task counts, one fixed provenance relation policy, frozen text embeddings, and node-ranking only.

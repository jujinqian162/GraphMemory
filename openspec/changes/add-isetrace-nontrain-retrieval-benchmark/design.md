## Context

M1/M2 expose `CanonicalTrajectory -> ProvenanceGraph -> MotifSpec -> ProvenanceQueryExample`, but deliberately stop before experiment integration. The experiment workflow currently assumes one raw file per split, requires train/dev/test configuration, supports only evidence-retrieval datasets, and passes query-conditioned `EvidenceGraph` objects to trainable graph methods. ISETrace instead needs two immutable inputs per test split (queries and trajectories), one trajectory graph reused by multiple queries, and a graph method that consumes the query-independent `ProvenanceGraph` directly.

The 100-query pilot references 61 graphs. Candidate ToolOutput counts range from 4 to 86 (median 24). All records remain LLM-authored and unreviewed; the assistant audit found 63 usable, 23 requiring revision, and 14 requiring rejection or regeneration. The implementation may exercise all structurally valid records under an explicit pilot policy but must not silently promote them to final gold.

## Goals / Non-Goals

**Goals:**

- Run BM25, frozen Dense, title-free entity GraphRAG, and one training-free provenance method over identical ToolOutput candidates.
- Keep physical provenance topology query-independent and label-blind.
- Use one content-addressed prepare/rank/evaluate workflow with deterministic task/graph joins.
- Make review policy and evidence-target policy explicit in config and artifact provenance.
- Reuse existing retrieval/evaluation contracts where scientifically valid and add typed provenance contracts where they are not.
- Preserve exact Dense fallback for GraphRAG and provenance-path retrieval when no valid intervention exists.

**Non-Goals:**

- Train Dense, R-GCN, relation weights, or query routers.
- Chunk ToolOutput candidates or change gold IDs from output level.
- Run full Microsoft GraphRAG with LLM extraction/community summaries.
- Treat assistant audit output as human annotation.
- Tune frozen heuristic parameters on the natural-query test pilot.
- Update paper results before manual review and formal experiment execution.

## Decisions

### 1. Use ToolOutput-level candidates with one shared renderer

Every query ranks all `execution.tool_output` nodes in its referenced trajectory. Candidate IDs remain provenance output IDs. Candidate text contains the native tool name, canonical arguments, and complete output text; every method receives the same text. Call and artifact nodes may be graph connectors but never ranking candidates.

Alternative: chunk long outputs. Rejected for this increment because labels identify outputs, not chunks, and aggregation would introduce another learned or heuristic choice.

### 2. Keep query/trajectory inputs independently content-addressed

An ISETrace split records a query source, trajectory source, and pinned source revision. The Prefect prepare task receives both `FileSourceRef` values so a change to either input invalidates cache identity. Prepared artifacts contain query tasks/labels plus unique provenance graphs; graph fingerprints remain independent of query text and labels.

Train/dev splits become optional at the generic config boundary. Test is always required, and trainable methods fail fast unless both train and dev are configured.

### 3. Make review and label policies explicit

`review_policy=allow_unreviewed` admits `unreviewed`, `accepted`, and `edited` records but never `rejected`; it is pilot-only. `review_policy=accepted_only` admits only `accepted` and `edited` records and is required for formal results.

`label_policy=intent_aware` uses `support_output_ids` for `complete_chain` and `contributing_sources`, and `answer_output_ids` for directional intents. `answer_only` and `support` remain explicit diagnostic policies. Gold dependency edges are retained only when both endpoints belong to the selected gold evidence set.

### 4. Derive one query-independent logical dependency view

A graph-domain projector derives output-to-output dependencies without reading queries or labels:

- `data.feeds`: source output -> consuming call -> returned target output;
- `resource.flow`: latest prior writer output -> later reader output through one explicit artifact.

Each logical dependency retains its namespaced relation, endpoint IDs, and supporting physical edge IDs. Motif extraction and training-free retrieval consume this same view rather than duplicating lifecycle logic. `temporal.precedes` alone never creates an evidence dependency.

### 5. Generalize GraphRAG only when titles are absent

The existing title/body entity bridge behavior is unchanged for document datasets. If no title groups exist, GraphRAG forms groups from entities shared by at least two candidate texts. Deterministic entity extraction includes capitalized spans and structured URL/path/filename tokens. It reads candidate text only and cannot access `ProvenanceGraph`, tool-call IDs as edges, or gold labels.

Dense seeds, frozen resolver, hub suppression, stable insertion, trace evidence, and exact fallback remain method-owned.

### 6. Add `provenance_path` as the training-free graph method

The method uses the same frozen Dense ranker as the Dense baseline. It takes the original Dense top-S outputs as anchors and searches the logical output dependency view in both directions because the hidden query intent is unavailable and natural queries ask for both upstream and downstream evidence.

Search is bounded by maximum logical hops, per-anchor partner count, total expansions, and a protected Dense prefix. Proposals require a schema-derived dependency path; temporal adjacency is insufficient. Ties use hop count, relation sequence, original anchor/partner rank, and node ID. Accepted partners are stably inserted after their anchor while preserving the relative order of non-promoted candidates and reusing the original descending Dense score slots.

Selected logical dependencies whose endpoints remain in top-k are emitted as `feeds` evaluation edges; their true namespaced relations and physical support are retained in a closed native trace. If no proposal causes an effective move, ranking IDs and scores are exactly Dense.

Alternative: fixed weighted diffusion. Deferred as a later diagnostic because fixed propagation weights are harder to justify and can spread through temporal noise.

### 7. Separate method input graphs from evaluation graphs

`provenance_path` consumes full query-independent `ProvenanceGraph` objects. Evaluation uses a task-local output-only `EvidenceGraph` projection containing all candidates and all query-independent logical dependencies, collapsed to the shared `feeds` edge category. BM25, Dense, and GraphRAG do not consume this evaluation graph, but all methods are evaluated against the same topology.

Execution-provenance evaluation marks query-to-evidence connectivity unsupported because no query-conditioned graph edges are created. Support, connectivity among evidence outputs, path, and edge metrics remain available.

### 8. Keep formal reporting gated

The committed pilot config explicitly names `allow_unreviewed` and `intent_aware`. Artifacts record both policies and the prompt/revision provenance. Documentation labels its metrics as engineering smoke/pilot results. Formal paper commands require an accepted/edited query file and `accepted_only`.

## Risks / Trade-offs

- **Long outputs are truncated by the frozen encoder.** BM25 still sees full text; record length statistics and defer output chunking to a controlled ablation.
- **Title-free entity extraction can create noisy hubs.** Preserve document-frequency gates, bounded partner proposals, and exact fallback; report bridge activation/fallback rates.
- **Bidirectional path search can over-complete local neighborhoods.** Exclude temporal-only edges, cap hops/partners/expansions, preserve a Dense prefix, and expose every proposal in the trace.
- **Intent-aware labels mix answer-event and complete-support tasks.** Persist intent per task and report complete-chain/contributing-source subsets separately; never interpret the aggregate alone as complete-support performance.
- **Prepared artifacts may duplicate candidate text across queries.** The pilot is small; unique provenance graphs remain stored once. A normalized candidate-set payload can replace duplicated task candidates later without changing public retrieval requests.

## Migration Plan

1. Add OpenSpec artifacts and focused contracts/tests without changing existing dataset behavior.
2. Add ISETrace benchmark preparation and logical dependency projection.
3. Integrate BM25/Dense and execution-provenance evaluation.
4. Add title-free GraphRAG fallback and activation diagnostics.
5. Add `provenance_path`, typed trace, registry/config/workflow integration.
6. Run pilot dry-runs with injected test encoders, then local CPU BM25 and available frozen Dense checks.
7. Run full quality gates; mark tasks complete but leave empirical paper claims pending.

Rollback removes the new dataset/method/config branches; existing evidence datasets and methods retain their prior contracts and cache identities.

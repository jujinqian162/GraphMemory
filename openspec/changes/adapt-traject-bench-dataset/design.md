## Context

The current dataset boundary assumes each raw split is one file and each prepared task owns a small candidate set. TRAJECT-Bench instead publishes one directory containing domain tool catalogs and 5,870 public queries: 2,000 parallel/simple, 2,000 parallel/hard, and 1,870 sequential queries. A query labels an ordered list of tool calls, while retrieval candidates must come from the separate public catalog. The repository's experiment runtime ranks typed text candidates and can optionally build EvidenceGraphs, so the adaptation should translate tool selection into those existing request contracts rather than importing the upstream LLM and tool-execution stack.

The primary pinned Hugging Face revision is `fbd4151a4897c4115679e184bf6c427d9d90e955`. At that revision, catalog files contain duplicate tool names and 64 queries reference at least one tool name absent from the corresponding public catalog. Its catalog adds optional `category` metadata and some query files encode the tool array as JSON text. The supported official GitHub snapshot `2723fd890778dbfb6af9e3aa8ee1c22272979468` exposes the older `public_data` layout and has 573 missing-catalog queries. Reconstructing missing candidates from either source's gold tool list would leak labels into ranking input.

## Goals / Non-Goals

**Goals:**

- Parse both parallel and sequential official JSON shapes into dataset-owned immutable records.
- Produce ranking artifacts that contain only query text and catalog-derived tool candidates, plus label artifacts that preserve the gold distinct tool set, repeated call sequence, and sequential dependencies.
- Reuse `TextRankingRequest`, `EvidenceGraphBuildRequest`, `ExecutionProvenanceRankingRequest`, `EvidenceEvaluationRequest`, and the registry-driven experiment workflow.
- Make directory raw inputs first-class plan artifacts and keep cache/status identity exact.
- Provide a pinned, copy-pasteable server path from download through BM25/Dense smoke and quick baselines.
- Keep invalid-upstream filtering deterministic, visible, and free of label-derived candidate repair.

**Non-Goals:**

- Executing RapidAPI tools, running ReAct/direct/CoT agents, calling an LLM judge, or reproducing final-answer accuracy.
- Claiming that retrieval ranking order is a predicted tool-execution trajectory.
- Training on an official TRAJECT-Bench training split; the benchmark publishes evaluation partitions, so the workflow split names are an operational mapping only.
- Adding the upstream repository or Hugging Face libraries as project runtime dependencies.

## Decisions

### 1. Dataset-owned operational partition mapping

`train` discovers `parallel/*/simple_ver.json`, `dev` discovers `parallel/*/hard_ver.json`, and `test` discovers `sequential/*/traj_query.json` plus `sequential/Travel/simple_ver.json`. The mapping is fixed in the TRAJECT-Bench prepare adapter and documented as operational, not official train/dev/test semantics. Files and records are ordered by normalized relative path and source index before seeded sampling, making Linux and Windows selection identical.

Alternative: materialize three aggregate JSON files after download. Rejected because it introduces a second data-conversion command and an untracked intermediate format before the repository's real prepare stage.

### 2. Candidate pools come only from public domain catalogs

Each query receives the canonicalized `tools/<Domain>_tool.json` pool. Catalog entries are keyed by exact tool name, duplicate entries are merged deterministically, connected-tool names are unioned, and candidate IDs are stable hashes of exact names with collision validation. Candidate text combines catalog name, provider/API identity, descriptions, and parameter schema; it never consumes query-local tool descriptions, parameter values, execution outputs, or final answers.

Queries whose gold tool names are absent from the canonical domain catalog are invalid and are filtered before seeded sampling. The prepare summary records raw, valid, dropped, duplicate-catalog, and missing-gold counts. This intentionally favors a smaller fair retrieval set over artificially perfect candidate coverage.

Alternative: use `all_tools.json`. Rejected for the initial baseline because the official domain-pool regime is faster and the global file has the same missing-gold problem plus substantially larger prepared artifacts. A future explicit pool-mode change can add it without changing the record boundary.

Alternative: add gold tool definitions into the catalog. Rejected as label leakage.

### 3. Distinct tool retrieval and repeated-call preservation are separate

The ranking label stores distinct `gold_tool_ids` in first-occurrence order because a retriever ranks tool APIs, not call instances. It also stores `gold_tool_sequence_ids` with repeats so no source trajectory information is lost. Sequential dependency edges are derived from adjacent call IDs, excluding self-loops and duplicate edges; parallel queries have no gold dependency edges. The existing evaluation request consumes the distinct set and dependency edges, while repeated sequence metadata remains available for a future planner/agent evaluator.

### 4. Graph input uses catalog knowledge, never query labels

Tool candidates project to `tool_api` graph items with no synthetic sequence index. Catalog `connected tools` links that resolve inside the candidate pool become directed `sequential` input-visible edges. Generic lexical/entity/query-overlap rules may add other graph edges. Gold order is projected only into `EvidenceLabel`, so graph-aware methods cannot see the answer during graph construction.

### 5. Raw source kind is explicit dataset configuration

`DatasetConfig` and its resolved form gain a required `source_kind` (`file` or `directory`). Existing datasets declare `file`; TRAJECT-Bench declares `directory`. The planner binds the external raw artifact using that value, allowing status validation to use `is_file()` or `is_dir()` without dataset-name conditionals. The prepare stage continues receiving a `Path`; only the dataset-owned script interprets directory contents.

### 6. Baseline scope and metrics

The initial runnable configuration targets frozen BM25, Dense, and execution-provenance retrieval. Existing Recall, Evidence F1, Full Support, and MRR metrics are semantically tool-retrieval metrics for this adapter; documentation maps those names explicitly and avoids presenting them as TRAJECT-Bench's end-to-end Exact Match, Usage, Trajectory Satisfaction, or Solution Accuracy. The recommended baseline uses `top_k=10`, matching the repository's current metric table. GraphRAG can also run through the same text request, but it is not required for the fastest server smoke.

The execution-provenance view is prospective and catalog-owned: each candidate API is represented as a `tool_call` node because it is a callable action, and only public catalog connections become `depends_on` edges. It does not claim these candidates were executed. Query-local calls, parameters, outputs, final answer, and gold order remain label-only. This gives the method a truthful, leakage-safe dependency graph while preserving the same offline tool-selection task as BM25 and Dense.

The experiment stage contract includes a dedicated `ExecutionProvenanceRetrieveStageConfig`. It builds dataset-owned provenance requests directly from prepared inputs, passes them through `ExecutionProvenanceBuildPayload`, and never schedules EvidenceGraph construction, pairs, or training for this stateless method. Supporting the method only in its direct registry builder is insufficient because public experiment selection must be end-to-end runnable.

## Risks / Trade-offs

- [Prepared artifacts repeat a domain catalog for every selected query] → Keep the initial pool domain-scoped, store one prebuilt candidate text per entry, measure real artifact size in the official-data smoke, and defer a shared-catalog artifact redesign until scale evidence requires it.
- [Upstream data changes can alter valid capacities] → Pin the documented Hugging Face revision, validate requested counts against the post-filter valid list, and report exact dropped reasons in every prepare summary.
- [Operational split names could be mistaken for official benchmark splits] → Put the mapping and the no-official-train warning in config comments, operations docs, and prepare metadata.
- [Catalog connected edges are incomplete or noisy] → Treat them as input-visible optional graph knowledge, skip unresolved endpoints, never repair them from gold sequences, and keep BM25/Dense baselines graph-independent.
- [Hash-based IDs are opaque] → Preserve exact tool names/provider/API in candidates and labels and fail on any detected hash collision.

## Migration Plan

1. Add explicit `source_kind: file` to existing dataset configs and propagate it through resolved plans without changing existing source paths.
2. Add the TRAJECT-Bench package, validation dispatch, prepare script, and dataset config.
3. Run synthetic contract tests, then prepare smoke records from the pinned official dataset.
4. Run BM25 and Dense workflow smoke from the standard experiment runner and inspect produced metrics/artifacts.
5. Roll back by removing the new dataset config/package and `source_kind` field changes; existing prepared artifacts remain independent JSON files.

## Open Questions

None for the initial offline retrieval adaptation. Global-pool retrieval and end-to-end tool execution require separate changes because they alter benchmark regime and runtime authority.

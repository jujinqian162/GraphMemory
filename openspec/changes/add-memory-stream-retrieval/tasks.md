## 1. Importance Data Contract and Cleaning

- [x] 1.1 Define compact schema-versioned importance contracts.
- [x] 1.2 Keep query-independent content digests and node-id extraction.
- [x] 1.3 Validate full artifact order, ids, digests, score range, and coverage.
- [x] 1.4 Support subset selection with duplicate and stale-content rejection.
- [x] 1.5 Add strict legacy cleaning with first-1000 canonical defaults.
- [x] 1.6 Implement task-local unique-level rank normalization with ties,
  half-up rounding, constant value `5`, and idempotence.
- [x] 1.7 Write compact output and statistics summary atomically.
- [x] 1.8 Remove annotation CLI, prompt, cache, runtime, config, and tests.

## 2. Memory Stream Retrieval

- [ ] 2.1 Add method tests with a fake dense seed ranker for relevance,
  request-owned recency, cleaned importance, complete-node output,
  constant signals, weighted sums, and node-id tie-breaks.
- [ ] 2.2 Add a Memory Stream-owned task-local min-max normalizer that maps
  constant signals to `0.0`.
- [ ] 2.3 Implement `MemoryStreamMethod` with an injected dense seed ranker and
  prevalidated `task_id -> TaskImportanceRecord` index; perform no file IO.
- [ ] 2.4 Add settings validation for non-negative weights, at least one
  positive weight, and `0 < recency_decay <= 1`.

## 3. Registry and Workflow

- [ ] 3.1 Add `RetrievalMethodId.MEMORY_STREAM`,
  `MemoryStreamRetrievalSettings`, `MemoryStreamBuildPayload`, and
  `ImportanceArtifactProvenance`.
- [ ] 3.2 Build the dense seed ranker through the existing dense contract,
  select/validate current-task importance records once, and inject both into
  `MemoryStreamMethod`.
- [ ] 3.3 Register Memory Stream with `RetrievalLifecycle.STATELESS` and reuse
  `STATELESS_RETRIEVAL_WORKFLOW`; add no workflow id or artifact role.
- [ ] 3.4 Add optional `RetrieveIO.importance`; set it only for Memory Stream
  from the `memory_stream_importance_path` experiment setting and default it to
  `data/hotpotqa/processed/memory_stream/dev.first_1000.importance.json`.
- [ ] 3.5 Load the compact artifact once in `run_retrieval.py`, before method
  construction and per-task timing; fail with the concrete missing path.
- [ ] 3.6 Add a Memory Stream-only test cap in workflow planning so cloud-full
  can warn and truncate to the available cleaned prefix instead of failing
  halfway through the run.
- [x] 3.7 Support `split_sources.dev/test = "importance"` for Memory Stream
  experiments by materializing run-local prepare artifacts from canonical
  processed input/labels joined in cleaned-importance task-id order.
- [x] 3.8 Validate importance-backed split materialization with missing task,
  stale digest, count/offset, unsupported method, and run-local artifact tests.
- [x] 3.9 Add `configs/experiments/hotpotqa_memory_stream.json` with
  `bm25`, `dense`, `memory_stream`, full 1000-task Memory Stream coverage, and
  `split_sources.dev/test = "importance"`.
- [ ] 3.10 Extend retrieval provenance with importance path, SHA-256, and schema
  version, and serialize weights, decay, capped test count, and encoder
  settings.
- [ ] 3.11 Add registry, stage-config, workflow, missing/stale artifact, default
  path, override path, capped-test warning, and run-summary provenance tests.

## 4. Verification

- [ ] 4.1 Run focused importance cleaning, method, registry, workflow, and
  provenance tests.
- [ ] 4.2 Run Ruff and BasedPyright.
- [ ] 4.3 Run the repository test suite.
- [ ] 4.4 Run real first-1000 cleaning and verify 1000 tasks / 41185 scores.
- [ ] 4.5 Run `openspec validate add-memory-stream-retrieval --strict`.

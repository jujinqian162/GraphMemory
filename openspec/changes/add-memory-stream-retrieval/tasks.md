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

- [ ] 2.1 Add tests for relevance, pseudo-recency, importance, normalization,
  constant signals, weights, and node-id tie-breaks.
- [ ] 2.2 Implement task-local score normalization.
- [ ] 2.3 Implement `MemoryStreamMethod` with equal default weights and
  `recency_decay=0.99`.
- [ ] 2.4 Reject invalid settings and invalid importance coverage.

## 3. Registry and Workflow

- [ ] 3.1 Add Memory Stream retrieval settings and builder.
- [ ] 3.2 Declare dense encoder plus read-only importance dependencies.
- [ ] 3.3 Add the workflow without an importance stage or run-local artifact.
- [ ] 3.4 Default to
  `data/hotpotqa/processed/memory_stream/dev.first_1000.importance.json`.
- [ ] 3.5 Record importance path/hash, weights, decay, and encoder provenance.

## 4. Verification

- [ ] 4.1 Run focused importance cleaning and retrieval tests.
- [ ] 4.2 Run Ruff and BasedPyright.
- [ ] 4.3 Run the repository test suite.
- [ ] 4.4 Run real first-1000 cleaning and verify 1000 tasks / 41185 scores.
- [ ] 4.5 Run `openspec validate add-memory-stream-retrieval --strict`.

## 1. Batch 7 - Training Pair Domain

- [x] 1.1 Add focused architecture/import tests that fail while callers depend on `graph_memory.learned.data` and while pair generation lacks an owned `training_pairs` entry
- [x] 1.2 Create `graph_memory/training_pairs/` modules for config, builder, samplers, and public exports
- [x] 1.3 Move train pair generation into `TrainPairBuilder` with independent negative samplers while preserving sampler order, random state, de-duplication, truncation, artifact rows, and summary statistics
- [x] 1.4 Update `scripts/build_train_pairs.py`, model training imports, and tests to use `graph_memory.training_pairs`
- [x] 1.5 Verify pair artifact focused tests and residual import searches for `graph_memory.learned.data`

## 2. Batch 8 - Trainable Graph Retriever Model Domain

- [x] 2.1 Add focused architecture/import tests that fail while model inference depends on training and while scripts/tests import trainable model internals from `graph_memory.learned`
- [x] 2.2 Create `graph_memory/models/graph_retriever/` modules for contracts, text embeddings, internals, config, factory, checkpoint, dev evaluation, training, inference, and public exports
- [x] 2.3 Move tensorization, batching, feature construction, neural components, checkpoint IO, model defaults, model factory, training loop, dev evaluation, and inference into the model domain without changing math or schema
- [x] 2.4 Move the checkpoint-backed retrieval adapter into `graph_memory/retrieval/methods/trainable_graph.py` and update retrieval factory/request wiring
- [x] 2.5 Update `scripts/train_graph_retriever.py`, `scripts/run_trainable_retrieval.py`, retrieval factory imports, and trainable graph tests to use owned model/retrieval modules while preserving parser contracts
- [x] 2.6 Verify tensorization, model forward, checkpoint round-trip, one-step training, trainable retrieval, parser contract, and residual import boundary tests

## 3. Final Verification

- [x] 3.1 Run focused Change C test suite, full pytest, type checking at error level, ruff, strict OpenSpec validation, and residual old-import searches

## Why

The project needs a trainable dense baseline that can reuse the existing `train_pairs.json` supervision without duplicating dense retrieval logic. The current train stage and CLI are shaped around the R-GCN graph retriever, so adding `dense_ft` now also exposes the need for method-specific training configs and one canonical train entry point.

## What Changes

- Add `dense_ft` as a public retrieval method alongside `dense` and `dense_rgcn_graph_retriever`.
- Train `dense_ft` by fine-tuning a SentenceTransformer from existing task, label, and train-pair artifacts.
- Reuse the existing `DenseTaskRetriever -> DenseEncodingService` inference path by loading the fine-tuned SentenceTransformer model directory.
- Add dense-owned query/passage formatting helpers shared by frozen dense inference and dense-ft training data construction.
- Add dense-ft model metadata so retrieval can recover training-time prefixes and encoder batch size from the checkpoint directory.
- Refactor TRAIN stage configs to method-specific root variants instead of forcing every training method into the R-GCN IO/dependency shape.
- **BREAKING**: replace `scripts/train_graph_retriever.py` with the unified `scripts/train_method.py --method <method>` entry point; no compatibility shim is retained.
- Add workflow, manifest, experiment-config, and documentation support for dense-ft smoke/quick/full runs.

## Capabilities

### New Capabilities

- `dense-finetune-training`: Building SentenceTransformers training/evaluation inputs from graph-memory artifacts and saving a reusable fine-tuned dense model directory.
- `method-specific-train-stage`: Loading, dispatching, and running train stage configs through method-specific root config variants and one canonical train script.
- `dense-ft-retrieval-workflow`: Planning, manifesting, retrieving, evaluating, and aggregating the `dense_ft` method through the experiment workflow.

### Modified Capabilities

None.

## Impact

- Affected code: dense embedding helpers, dense-ft model package, retrieval method registry/builders, training registry, TRAIN stage config conversion, train stage runner, workflow artifacts/manifests/stage projections, experiment config, and operations docs.
- Affected CLI: train commands now use `scripts/train_method.py --method <method>` for both R-GCN and dense-ft.
- Affected artifacts: dense-ft checkpoint is a SentenceTransformer model directory under the learned checkpoint role, not a `.pt` file.
- Dependencies: pin `sentence-transformers==2.7.0` and use its native `InputExample`, `DataLoader`, and `SentenceTransformer.fit()` training API.
- Verification: focused dense-ft data/training/retrieval/workflow tests, R-GCN regression tests, experiment runner tests, basedpyright, and OpenSpec validation.

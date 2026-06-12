## Context

The repository already has a shared frozen dense inference boundary: `DenseTaskRetriever` depends on `DenseEncodingService`, and the dense service owns sentence encoder invocation, normalization, batching, and query/passage text construction. R-GCN training, however, is still represented by train-stage and script contracts that assume graph tensors, graph feature providers, seed-signal providers, and `.pt` checkpoints.

`dense_ft` needs the same public experiment lifecycle as other retrieval methods, but its training input is task/label/pair artifacts and its checkpoint is a SentenceTransformer model directory. The implementation must therefore avoid creating a second dense inference path while making TRAIN stage dispatch precise enough for non-graph train methods.

## Goals / Non-Goals

**Goals:**

- Introduce `dense_ft` as a public retrieval method with experiment workflow support.
- Fine-tune a SentenceTransformer from existing `train_pairs.json` supervision.
- Reuse the current dense inference implementation by pointing `DenseConfig.model_name` at the saved model directory.
- Share dense query/passage formatting between frozen dense inference and dense-ft training data builders.
- Represent TRAIN configs as method-specific root variants and dispatch through the training registry.
- Replace the R-GCN-specific train script with one canonical `scripts/train_method.py --method <method>` entry.
- Keep profile files as overrides of defaults, with profile entries containing only values that differ from defaults.

**Non-Goals:**

- Do not introduce a new dense scoring implementation.
- Do not put dense-ft training code under the R-GCN graph retriever package.
- Do not add persistent embedding caches, distributed SentenceTransformer training, DDP orchestration, or cross-encoder training.
- Do not keep `scripts/train_graph_retriever.py` as a compatibility shim.
- Do not change `dense` frozen retrieval behavior or its artifact schema.

## Decisions

### Decision: `dense_ft` is a public method, not a `dense` profile

`dense_ft` gets its own `RetrievalMethodId`, workflow mapping, training config, and experiment table entry. It is still implemented as dense retrieval at inference time, but the method identity matters because it has a learned checkpoint/model directory and a train stage.

Alternative considered: model `dense_ft` as `dense` plus a different encoder path. Rejected because the experiment workflow, manifest, and result tables need to distinguish frozen dense from fine-tuned dense.

### Decision: Inference reuses `DenseTaskRetriever`

The retrieval builder for `dense_ft` reads `<checkpoint>/dense_ft_model_config.json`, constructs a `DenseConfig` with `model_name=str(checkpoint)`, and returns `DenseTaskRetriever`. Prefixes, normalization, batching, score formula, and ranking tie-breaks stay dense-owned.

Alternative considered: add a dense-ft retriever class. Rejected because it would duplicate ranking behavior and increase the risk that frozen dense and dense-ft diverge.

### Decision: Dense text formatting lives in `graph_memory.embeddings.dense`

Add `format_dense_query()` and `format_dense_passage()` helpers and make `DenseEncodingService` and dense-ft data builders call them. The dense-ft package must not hand-roll `source/text` formatting.

Alternative considered: keep formatting private to `DenseEncodingService`. Rejected because dense-ft training needs exactly the same text contract before model training begins.

### Decision: SentenceTransformer training package owns dense-ft data and trainer integration

Create `graph_memory.models.dense_finetune` with contracts, data builders, and training orchestration. Data builders convert task/label/pair artifacts into train rows and IR evaluator payloads. Training code writes both the SentenceTransformer model directory and `dense_ft_model_config.json`.

Alternative considered: implement dense-ft inside scripts. Rejected because scripts should adapt artifacts and delegate method behavior to registry/trainer code.

### Decision: Dense-ft uses only the SentenceTransformers 2.7.0 training API

The implementation pins `sentence-transformers==2.7.0` and uses one training path:
`InputExample` rows are loaded by a PyTorch `DataLoader`, paired with
`MultipleNegativesRankingLoss`, evaluated by `InformationRetrievalEvaluator`,
and trained through `SentenceTransformer.fit()`.

The project does not import `SentenceTransformerTrainer`,
`SentenceTransformerTrainingArguments`, `datasets`, or `accelerate` for
dense-ft training. It also does not keep a version detector or fallback path.
Trainer-only configuration fields are removed; the dense-ft config exposes the
parameters consumed by the 2.7.0 path.

Alternative considered: support both 2.7.0 and newer Trainer APIs. Rejected
because the deployment environment is fixed to a vendor-adapted Python 3.10
stack, and two backends would add unnecessary runtime branching and divergent
semantics.

### Decision: TRAIN stage uses root-level method-specific config variants

`TrainStageConfig` becomes a discriminated union of `RgcnTrainStageConfig` and `DenseFinetuneTrainStageConfig`. Each root carries the method id, precise IO shape, and method settings. The config loader remains the single public train config entry.

Alternative considered: add optional fields to one train config dataclass. Rejected because it recreates optional-bag semantics and forces dense-ft to carry meaningless graph fields.

### Decision: The train script is unified and breaking

All workflow-generated train commands use `scripts/train_method.py --method <method>`. Direct train commands must pass `--method dense_rgcn_graph_retriever` or `--method dense_ft`. The old `scripts/train_graph_retriever.py` is deleted rather than retained as a wrapper.

Alternative considered: keep a compatibility shim. Rejected because the user explicitly wants one canonical train entrance and no long-lived split between old and new adapters.

### Decision: Profiles are override-only

`configs/training/dense_ft/base.json` stores common CUDA defaults under `defaults`; profiles only contain fields that differ from those defaults. For example, `smoke` may override `device` to `cpu`, but `quick` and `cloud-full` must not repeat `device: "cuda"` if that is already the default.

Alternative considered: expand every profile to complete effective config. Rejected because this repo treats profiles as overrides, and repeated defaults obscure the actual variation being tested.

## Risks / Trade-offs

- [TRAIN refactor touches R-GCN regression paths] -> Keep the union change localized to TRAIN configs, registry dispatch, stage runner, and train script; run R-GCN training tests after each stage.
- [SentenceTransformers evaluator returns its main score directly] -> Use the 2.7.0 `cos_sim` score function and record its returned MAP@100 value as `eval_dev_cos_sim_map@100`.
- [Local dependency drift] -> Pin `sentence-transformers==2.7.0` and update `uv.lock`.
- [Checkpoint role name still says checkpoint while dense-ft uses a directory] -> Keep the artifact role for workflow compatibility, but document and validate that dense-ft checkpoint is a SentenceTransformer model directory.
- [Full train pairs can be large] -> Limit negatives per positive in config, defaulting to one hard negative per positive.

## Migration Plan

1. Create OpenSpec artifacts for dense-ft and validate them.
2. Add failing tests for dense formatting and dense-ft data construction, then implement the shared helpers and data package.
3. Add dependencies, dense-ft training contracts, fake-trainer tests, and real SentenceTransformers trainer wiring.
4. Refactor TRAIN config, registry dispatch, and the train script; delete the old R-GCN-specific train script and update references.
5. Register dense-ft retrieval and workflow support, then update experiment config and docs.
6. Run focused tests, broader regression tests, type checks, experiment planning smoke, and strict OpenSpec validation.

Rollback is a normal code revert plus removal of the new OpenSpec change directory. Existing frozen dense and R-GCN artifacts do not require data migration.

## Open Questions

None. The first implementation intentionally uses `MultipleNegativesRankingLoss` and a bounded number of hard negatives per positive.

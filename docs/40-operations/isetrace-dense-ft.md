# ISETrace Dense-FT candidate-view controls

`dense_ft` exposes `variant=flat|provenance_unit` on ISETrace. Both variants fine-tune the same E5 encoder with the same trainer and scientific settings; only the prepared candidate view changes. `flat` remains the default.

## Scientific boundary

- `flat` ranks the same lossless trajectory chunks used by BM25, flat Dense, and GraphRAG.
- `provenance_unit` ranks the source-backed argument/output content units used as candidates by provenance methods, but does not receive their graph.
- Natural-query gold `SourceSpan` values are mapped to every overlapping candidate in the selected view.
- Dense-FT receives query text, candidate text, and materialized text pairs only.
- Neither variant receives `ProvenanceGraph`, `EvidenceGraph`, motifs, dependency edges, graph/node IDs, or native graph traces.
- ISETrace forces the effective `hard_graph_neighbor_per_positive` to `0`; easy-random, BM25-hard, and Dense-hard negatives remain configurable.
- Dev ranking is task-local: each query is compared only with chunks from its own trajectory, and the best checkpoint is selected by natural-query Recall@5.
- Training batches contain at most one example from each trajectory to avoid trajectory-local in-batch false negatives.
- Pair artifacts, model artifacts, run results, and checkpoint metadata record the variant. Retrieval rejects a checkpoint trained for the other view; legacy metadata without the field is flat.
- Test is fixed to the first 2,000 resolvable natural queries in authoring-file order (1,207 trajectories) and uses exact-span Recall, MRR, token-budget Coverage, Full Support, span F1, and evidence density. Coverage and Full Support are emitted at 256/512/1024/2048/4096/8192 tokens, together with trapezoidal Budget-AUC normalized on the log2-token axis, so complete budget curves can be reconstructed without rerunning retrieval. The content-addressed authoring metadata sidecar supplies `memory_mode`, which is preserved in per-task evaluation rows for stratification but never exposed to the retriever. Path/edge metrics remain unavailable.

## Smoke and full runs

Run the provenance-unit engineering smoke without changing the flat Dense-FT hyperparameters:

```bash
uv run python experiment/run.py \
  name=isetrace_dense_ft_provenance_unit_smoke \
  dataset=isetrace \
  method=dense_ft \
  method.variant=provenance_unit \
  profile=smoke \
  seed=13 \
  split_seed=13 \
  device=cuda:0
```

Run the matched full candidate-view pair as separate Hydra jobs:

```bash
uv run python experiment/run.py -m \
  name=isetrace_dense_ft_candidate_view_s13 \
  dataset=isetrace \
  method=dense_ft \
  method.variant=flat,provenance_unit \
  profile=full \
  seed=13 \
  split_seed=13 \
  device=cuda:0
```

Effective counts are:

```text
train: trajectories=2410 -> natural queries=3894
 dev: trajectories=352  -> natural queries=580
test: trajectories=1207 -> natural queries=2000
```

Trajectory selection includes every available authored query on each selected trajectory. Changing the model seed does not change the split because splitting uses `split_seed`.

## Formal protocol

Use seed 13 for one engineering pilot to verify the provenance-unit lifecycle, but do not tune its learning rate or epoch count separately: those values stay matched to flat Dense-FT. Then run both variants with unchanged configuration and model seeds `13`, `17`, and `29`; keep `split_seed=13`. Independent jobs may be assigned to separate visible GPUs; this baseline does not require single-model distributed training.

In paper tables, label the two rows by method plus variant (for example, `Flat-Chunk Dense-FT` and `Provenance-Unit Dense-FT`). A provenance-unit gain over flat Dense-FT estimates candidate-segmentation effects, not graph reasoning; only comparison against provenance path/R-GCN can attribute traversal or message-passing effects.

Generated natural queries are engineering data until separately reviewed and frozen. Do not add paper result tables from smoke or unreviewed pilot runs.

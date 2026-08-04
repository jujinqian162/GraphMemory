# ISETrace Dense-FT

`dense_ft` is the supervised flat-text baseline for revised RQ2. It fine-tunes the same E5 encoder used by frozen Dense and ranks the same lossless flat trajectory chunks used by BM25, Dense, and GraphRAG.

## Scientific boundary

- Natural and template gold `SourceSpan` values are mapped to every overlapping flat chunk.
- Dense-FT receives query text, chunk text, and materialized text pairs only.
- It does not receive `ProvenanceGraph`, EvidenceGraph, motifs, dependency edges, graph/node IDs, query origin, or native graph traces.
- ISETrace forces the effective `hard_graph_neighbor_per_positive` to `0`; easy-random, BM25-hard, and Dense-hard negatives remain configurable.
- Dev ranking is task-local: each query is compared only with chunks from its own trajectory.
- Mixed dev selects the best checkpoint by natural Recall@5; template-only dev uses template Recall@5.
- Training batches contain at most one example from each trajectory to avoid trajectory-local in-batch false negatives.
- Test is fixed to the first 2,000 resolvable natural queries in authoring-file order (1,207 trajectories) and uses exact-span Recall, MRR, token-budget Coverage, Full Support, span F1, and evidence density. Path/edge metrics remain unavailable.

## Default mixed run

The dataset default uses natural-only supervision for Dense-FT; template-ratio sweeps belong to provenance R-GCN:

```bash
python experiment/run.py \
  name=isetrace_dense_ft_nat1_tpl2_s13 \
  dataset=isetrace \
  method=dense_ft \
  profile=full \
  seed=13 \
  split_seed=13 \
  device=cuda:0
```

Effective counts are:

```text
train: natural trajectories=2410 -> natural queries=3894
 dev: natural trajectories=352  -> natural queries=580
test: natural trajectories=1207 -> natural queries=2000
```

## Natural-only control

```bash
python experiment/run.py \
  name=isetrace_dense_ft_natural_only_s13 \
  dataset=isetrace \
  method=dense_ft \
  profile=full \
  seed=13 \
  split_seed=13 \
  device=cuda:0 \
  dataset.trajectories.splits.train.template=0 \
  dataset.trajectories.splits.dev.template=0
```

Both commands use deterministic trajectory-count-based splits. Natural trajectory selection includes every available authored query on each selected trajectory; changing the model seed does not change the split because splitting uses `split_seed`.

## Formal protocol

Use seed 13 for one engineering pilot to freeze learning rate and epoch count. Then run the unchanged configuration with model seeds `13`, `17`, and `29`. Independent seeds may be assigned to separate visible GPUs; this baseline does not require single-model distributed training.

Generated natural queries are engineering data until separately reviewed and frozen. Do not add paper result tables from smoke or unreviewed pilot runs.

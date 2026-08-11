# ISETrace Dense-FT

`dense_ft` is the supervised flat-text baseline for ISETrace. It fine-tunes the same E5 encoder used by frozen Dense and ranks the same lossless flat trajectory chunks used by BM25, Dense, and GraphRAG.

## Scientific boundary

- Natural-query gold `SourceSpan` values are mapped to every overlapping flat chunk.
- Dense-FT receives query text, chunk text, and materialized text pairs only.
- It does not receive `ProvenanceGraph`, `EvidenceGraph`, motifs, dependency edges, graph/node IDs, or native graph traces.
- ISETrace forces the effective `hard_graph_neighbor_per_positive` to `0`; easy-random, BM25-hard, and Dense-hard negatives remain configurable.
- Dev ranking is task-local: each query is compared only with chunks from its own trajectory, and the best checkpoint is selected by natural-query Recall@5.
- Training batches contain at most one example from each trajectory to avoid trajectory-local in-batch false negatives.
- Test is fixed to the first 2,000 resolvable natural queries in authoring-file order (1,207 trajectories) and uses exact-span Recall, MRR, token-budget Coverage, Full Support, span F1, and evidence density. The content-addressed authoring metadata sidecar supplies `memory_mode`, which is preserved in per-task evaluation rows for stratification but never exposed to the retriever. Path/edge metrics remain unavailable.

## Full run

```bash
uv run python experiment/run.py \
  name=isetrace_dense_ft_natural_s13 \
  dataset=isetrace \
  method=dense_ft \
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

Use seed 13 for one engineering pilot to freeze learning rate and epoch count. Then run the unchanged configuration with model seeds `13`, `17`, and `29`. Independent seeds may be assigned to separate visible GPUs; this baseline does not require single-model distributed training.

Generated natural queries are engineering data until separately reviewed and frozen. Do not add paper result tables from smoke or unreviewed pilot runs.

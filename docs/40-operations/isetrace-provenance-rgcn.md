# ISETrace Provenance R-GCN

`provenance_rgcn` is the trainable execution-provenance method. It consumes the existing query-independent `ProvenanceGraph` directly and reuses the maintained R-GCN encoder, node scorer, graph batcher, BCE loss, AdamW loop, gradient clipping, dev selection, and checkpoint owner. It does not restore the deleted label-conditioned provenance implementation or convert provenance into `EvidenceGraph`.

## Configuration

`configs/dataset/isetrace.yaml` exposes only:

- `trajectory_source`
- `natural_query_source`
- explicit `trajectories.splits.<split>` trajectory counts
- content chunking settings

The source revision is inferred from dataset registration and checked against authoring run metadata when available. Split assignment uses `split_seed` and groups every query for one trajectory. Model seed changes do not move trajectories between splits.

Natural queries are resolved before allocation. Invalid or unresolvable records are excluded and counted. Train, development, and test select disjoint trajectory sets and keep every available authored natural query belonging to each selected trajectory.

## Supervision and negatives

Positives are provenance content candidates whose exact source spans overlap natural-query gold spans. Pair construction reuses easy-random, BM25-hard, Dense-hard, and graph-neighbor samplers over the provenance candidate universe. A task with no positive candidate fails before optimization.

`memory_mode` remains in the prepared query metadata and is copied into per-task evaluation rows for reproducible stratified analysis. It is not copied into retrieval requests, candidates, persisted graphs, embeddings, or numeric model features. The natural authoring metadata sidecar is a required content-addressed preparation input.

## Graph policy

The tensorizer appends an ephemeral disconnected `q` node. Persisted graph fingerprints do not change. Message passing uses uniform forward/reverse IDs for:

- `execution.returns`
- `execution.has_argument`
- `execution.has_content`
- `data.feeds`
- `resource.reads`
- `resource.writes`
- `content.next`

`temporal.precedes` and metadata-derived numeric features are excluded. `method.variant=wo_graph` sets the existing R-GCN layer count to zero and provides the candidate-matched no-message-passing control.

## Lifecycle

```text
prepare train/dev/test natural queries
  -> build provenance candidate pairs
  -> freeze train/dev graph + query embeddings
  -> train shared node-ranking R-GCN
  -> select on dev Recall@5
  -> save strict provenance_rgcn checkpoint
  -> reload checkpoint and rank test requests
  -> exact-span evaluation
```

Test reports exact-span Recall, Coverage@512/1024/2048 Tokens, Full Support, span F1, MRR, and evidence density. Candidate source spans are preserved through ranking.

There are no independently annotated provenance edge/path labels in the natural corpus. Therefore path/edge accuracy is unavailable; any retrieved provenance path is diagnostic only and must not be reported as labeled accuracy.

## Commands

```bash
uv run python experiment/run.py \
  name=isetrace_rgcn_smoke dataset=isetrace profile=smoke \
  method=provenance_rgcn device=cpu

uv run python experiment/run.py \
  name=isetrace_rgcn_natural_s13 dataset=isetrace profile=full \
  method=provenance_rgcn seed=13 split_seed=13 device=cuda:0
```

The full split contains 3,894 train queries, 580 development queries, and 2,000 test queries. Formal multi-seed evaluation repeats the unchanged configuration with model seeds 13, 17, and 29 while keeping `split_seed=13`.

## Claim boundary

Generated natural queries remain unreviewed until a separate review and freeze process is completed. Engineering smoke runs verify contracts and reproducibility only. Formal claims require the reviewed frozen natural test corpus and the complete seed set.

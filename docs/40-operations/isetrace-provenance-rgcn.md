# ISETrace Provenance R-GCN

`provenance_rgcn` is the trainable execution-provenance method. It consumes the existing query-independent `ProvenanceGraph` directly and reuses the maintained R-GCN encoder, node scorer, graph batcher, BCE loss, AdamW loop, gradient clipping, dev selection, and checkpoint owner. It does not restore the deleted label-conditioned provenance implementation or convert provenance into `EvidenceGraph`.

## Configuration

`configs/dataset/isetrace.yaml` exposes only:

- `trajectory_source`
- `natural_query_source`
- explicit `trajectories.splits.<split>.natural/template` trajectory counts
- content chunking settings

The source revision is inferred from dataset registration and checked against authoring run metadata when available. Split assignment uses `split_seed` and groups every query for one trajectory. Model seed changes do not move trajectories between splits.

Natural queries are resolved before allocation. Invalid or unresolvable records are excluded and counted. Train/dev select exactly their configured natural/template counts from frozen trajectory partitions, including valid `natural: 0` template-only runs. Test never contains templates.

## Supervision and negatives

Natural positives are provenance content candidates whose exact source spans overlap natural gold spans. Template positives are only content chunks attached through `execution.has_content` to focused ToolOutputs. Other motif participants are not positive and remain eligible negatives.

Pair construction reuses easy-random, BM25-hard, Dense-hard, and graph-neighbor samplers over the provenance candidate universe. A task with no positive candidate fails before optimization.

Query origin and `memory_mode` remain in prepared sidecars and are copied into per-task evaluation rows for reproducible stratified analysis. Motif identity and template audit fields remain preparation-only. None of these fields are copied into retrieval requests, candidates, persisted graphs, embeddings, or numeric model features. The natural authoring metadata sidecar is a required content-addressed preparation input.

## Graph policy

The tensorizer appends an ephemeral disconnected `q` node. Persisted graph fingerprints do not change. Message passing uses uniform forward/reverse IDs for:

- `execution.returns`
- `execution.has_argument`
- `execution.has_content`
- `data.feeds`
- `resource.reads`
- `resource.writes`
- `content.next`

`temporal.precedes` and metadata-derived numeric features are excluded. `method.variant=wo_graph` sets the existing R-GCN layer count to zero; it is the direct template-training control, not a separate model.

## Lifecycle

```text
prepare train/dev/test
  -> build provenance candidate pairs
  -> freeze train/dev graph + query embeddings
  -> train shared node-ranking R-GCN
  -> select on natural dev Recall@5, or template dev Recall@5 when dev is template-only
  -> save strict provenance_rgcn checkpoint
  -> reload checkpoint and rank natural-only test requests
  -> exact-span evaluation
```

Natural and template dev metrics are reported separately. Mixed dev selects on natural Recall@5; template-only dev selects on template Recall@5 and records that origin in the model artifact. Test reports exact-span Recall, Coverage@512/1024/2048 Tokens, Full Support, span F1, MRR, and evidence density. Candidate source spans are preserved through ranking.

There are no independently annotated provenance edge/path labels in the natural corpus. Therefore path/edge accuracy is unavailable; any retrieved provenance path is diagnostic only and must not be reported as labeled accuracy.

## Commands

```bash
uv run python experiment/run.py \
  name=isetrace_rgcn_smoke dataset=isetrace profile=smoke \
  method=provenance_rgcn device=cpu \
  dataset.trajectories.splits.train.natural=4 \
  dataset.trajectories.splits.train.template=4 \
  dataset.trajectories.splits.dev.natural=2 \
  dataset.trajectories.splits.dev.template=2 \
  dataset.trajectories.splits.test.natural=2

uv run python experiment/run.py \
  name=isetrace_rgcn_full dataset=isetrace profile=full \
  method=provenance_rgcn device=cuda:0

uv run python experiment/run.py \
  name=isetrace_rgcn_template_only dataset=isetrace profile=full \
  method=provenance_rgcn device=cuda:0 \
  dataset.trajectories.splits.train.natural=0 \
  dataset.trajectories.splits.train.template=8076 \
  dataset.trajectories.splits.dev.natural=0 \
  dataset.trajectories.splits.dev.template=786
```

The ratio sweeps use the fixed 3,894 train and 580 dev natural queries. Template counts are trajectory counts because each selected trajectory emits one template query. The full-pool setting uses all 21,746 eligible non-test template trajectories: 18,927 train and 2,819 dev, giving an aggregate ratio of `1:4.8605`.

## Claim boundary

Generated natural queries remain unreviewed until a separate review and freeze process is completed. Engineering smoke runs verify contracts and reproducibility only. Do not add paper result tables or formal claims until a reviewed frozen natural test corpus exists and formal seeds 13/17/29 have been run.

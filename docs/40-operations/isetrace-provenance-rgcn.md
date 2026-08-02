# ISETrace Provenance R-GCN

`provenance_rgcn` is the trainable execution-provenance method. It consumes the existing query-independent `ProvenanceGraph` directly and reuses the maintained R-GCN encoder, node scorer, graph batcher, BCE loss, AdamW loop, gradient clipping, dev selection, and checkpoint owner. It does not restore the deleted label-conditioned provenance implementation or convert provenance into `EvidenceGraph`.

## Configuration

`configs/dataset/isetrace.yaml` exposes only:

- `trajectory_source`
- `natural_query_source`
- normalized `queries.split_ratio`
- train/dev `queries.mix_ratio`
- content chunking settings

The source revision is inferred from dataset registration and checked against authoring run metadata when available. Split assignment uses `split_seed` and groups every query for one trajectory. Model seed changes do not move trajectories between splits.

Natural queries are resolved before allocation. Invalid or unresolvable records are excluded and counted. Train/dev retain all selected natural records and add a deterministic ratio-derived number of templates rendered from existing motifs. Test never contains templates.

## Supervision and negatives

Natural positives are provenance content candidates whose exact source spans overlap natural gold spans. Template positives are only content chunks attached through `execution.has_content` to focused ToolOutputs. Other motif participants are not positive and remain eligible negatives.

Pair construction reuses easy-random, BM25-hard, Dense-hard, and graph-neighbor samplers over the provenance candidate universe. A task with no positive candidate fails before optimization.

Query origin, motif identity, and template audit fields remain in prepared sidecars. They are never copied into retrieval requests, candidates, persisted graphs, embeddings, or numeric model features.

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
  -> select on natural dev Recall@5
  -> save strict provenance_rgcn checkpoint
  -> reload checkpoint and rank natural-only test requests
  -> exact-span evaluation
```

Natural and template dev metrics are reported separately. Template metrics are diagnostic; primary checkpoint selection is natural-only. Test uses the existing ISETrace span Recall/Coverage, Full Support, span F1, MRR, evidence-density, and token-budget metrics. Candidate source spans are preserved through ranking.

There are no independently annotated provenance edge/path labels in the natural corpus. Therefore path/edge accuracy is unavailable; any retrieved provenance path is diagnostic only and must not be reported as labeled accuracy.

## Commands

```powershell
uv run python experiment/run.py name=isetrace_rgcn_smoke dataset=isetrace profile=smoke method=provenance_rgcn device=cpu
uv run python experiment/run.py name=isetrace_rgcn_full dataset=isetrace profile=full method=provenance_rgcn device=cuda:0
uv run python experiment/run.py name=isetrace_rgcn_wo_graph dataset=isetrace profile=quick method=provenance_rgcn method.variant=wo_graph device=cuda:0
```

The `full` profile consumes the complete configured mixed train/dev splits rather than the evidence workflow's fixed dev cap.

## Claim boundary

Generated natural queries remain unreviewed until a separate review and freeze process is completed. Engineering smoke runs verify contracts and reproducibility only. Do not add paper result tables or formal claims until a reviewed frozen natural test corpus exists and formal seeds 13/17/29 have been run.

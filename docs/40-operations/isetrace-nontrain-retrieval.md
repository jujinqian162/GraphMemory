# ISETrace v7 non-training retrieval

This workflow evaluates the derived natural-only test split of the four-field v7 authoring corpus against pinned ISETrace trajectories. It does not build pairs or train a model.

## Input and gold

Each query record contains only `id`, handle-delimited `text`, `query`, and one or more `{source, quote}` gold entries. During preparation, the compiler:

1. scans the configured raw trajectory directory and uniquely matches the task text to one trajectory;
2. resolves each paired `A*` ToolCall and `E*` ToolOutput handle;
3. requires every quote to occur exactly once inside its named source;
4. converts the quotes directly into exact source-coordinate spans;
5. excludes and counts generated candidates that cannot be compiled to one exact raw event.

The prepared tasks, labels, selected trajectories, and provenance graphs are content-addressed artifacts under `data/processed/`; callers do not materialize an evaluable query file or a trajectory subset.

There is one gold interpretation: the complete set of exact spans in `gold`. V7 has no answer/support policy, output-ID gold, query-intent label, motif label, or dependency-edge gold. Consequently path/edge recall is `N/A` unless a separate independently annotated edge-label dataset is introduced.

## Retrieval views

Every method receives the same query and underlying trajectory.

- BM25, Dense with `variant=flat`, and GraphRAG rank the same token-window chunks of the rendered full trajectory, including messages, ToolCall arguments, and ToolOutputs.
- Dense with `variant=provenance_unit` ranks the prepared source-backed argument/output content units with the unchanged frozen encoder and cosine scorer. It receives only `TextRankingRequest` and never loads a provenance graph; this isolates candidate segmentation from path traversal.
- GraphRAG privately subdivides flat chunks into smaller graph-index text units, builds a noun-phrase co-occurrence graph, runs query-personalized PageRank, and projects graph scores back to the original token-window candidates. Its graph never receives provenance edges or gold spans.
- `provenance_path` ranks the same provenance units and may additionally traverse the query-independent provenance graph.
- The provenance graph may change retrieval order and the diagnostic retrieved subgraph, but it never creates or expands gold.

All candidates retain exact `(event_id, json_pointer, char_start, char_end)` coordinates. Evaluation unions these coordinates rather than comparing method-local candidate IDs.

## Commands

The dataset config consumes only the generated authoring file and the raw trajectory directory:

```text
data/isetrace/query-authoring/isetrace-v7-raw.jsonl
data/isetrace/raw/trajectories/
```

For another authoring file, override `dataset.natural_query_source`. `dataset.trajectory_source` remains the registered raw root or trajectory directory. `trajectories.splits.test.natural` selects test trajectories, and every valid authored query on those trajectories is evaluated. Non-training methods schedule only test preparation.

Smoke runs:

```bash
uv run python experiment/run.py name=isetrace_v7_bm25_smoke dataset=isetrace profile=smoke method=bm25 device=cpu
uv run python experiment/run.py name=isetrace_v7_dense_smoke dataset=isetrace profile=smoke method=dense device=cuda:0
uv run python experiment/run.py name=isetrace_v7_provenance_unit_dense_smoke dataset=isetrace profile=smoke method=dense method.variant=provenance_unit device=cuda:0
uv run python experiment/run.py name=isetrace_v7_graphrag_smoke dataset=isetrace profile=smoke method=graphrag device=cuda:0
uv run python experiment/run.py name=isetrace_v7_path_smoke dataset=isetrace profile=smoke method=provenance_path device=cuda:0
```

Full non-training runs should keep all method parameters frozen:

```bash
uv run python experiment/run.py -m \
  name=isetrace_v7_nontrain dataset=isetrace profile=full device=cuda:0 \
  method=bm25,dense,graphrag,provenance_path

uv run python experiment/run.py -m \
  name=isetrace_v7_dense_candidate_view dataset=isetrace profile=full device=cuda:0 \
  method=dense method.variant=flat,provenance_unit
```

The second command is the matched frozen candidate-view control. Both jobs retain method ID `dense`; the resolved `variant` in the run summary, manifest, and tracking data distinguishes the rows.

## Interpretation

Report span Coverage/Recall, Span F1, Full Support, Evidence Density, and MRR. The complete token-budget curves report Coverage and Full Support at 256/512/1024/2048/4096/8192 tokens, plus trapezoidal Budget-AUC normalized on the log2-token axis over the 256–8192 interval; the compact main table may continue to emphasize 512/1024/2048. Compare Dense `flat` against Dense `provenance_unit` to estimate the candidate-segmentation contribution; compare provenance-unit Dense against `provenance_path` to estimate path-traversal contribution. Graph connectivity can remain a method diagnostic, but it is not a labeled dependency metric in v7.

Experiment scale is determined from successfully compiled records rather than the filename. Preparation writes `queries_seen`, `queries_resolved`, `queries_dropped`, `queries_uncompilable`, `queries_unmatched`, and `queries_ambiguous` into the dataset artifact counts.

The generated query file is an authoring candidate set until separately reviewed and frozen. Review state is not embedded in the four-field records.

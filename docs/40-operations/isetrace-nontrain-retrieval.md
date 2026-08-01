# ISETrace v7 non-training retrieval

This workflow evaluates the four-field v7 authoring records against the pinned ISETrace trajectories. It does not train a model.

## Input and gold

Each query record contains only `id`, handle-delimited `text`, `query`, and one or more `{source, quote}` gold entries. During preparation, the compiler:

1. uniquely matches the task text to one trajectory;
2. resolves each paired `A*` ToolCall and `E*` ToolOutput handle;
3. requires every quote to occur exactly once inside its named source;
4. converts the quotes directly into exact source-coordinate spans.

There is one gold interpretation: the complete set of exact spans in `gold`. V7 has no answer/support policy, output-ID gold, query-intent label, motif label, or dependency-edge gold. Consequently path/edge recall is `N/A` unless a separate independently annotated edge-label dataset is introduced.

## Retrieval views

Every method receives the same query and underlying trajectory.

- BM25, Dense, and GraphRAG rank the same token-window chunks of the rendered full trajectory, including messages, ToolCall arguments, and ToolOutputs.
- GraphRAG privately subdivides those chunks into smaller graph-index text units, builds a noun-phrase co-occurrence graph, runs query-personalized PageRank, and projects graph scores back to the original token-window candidates. Its graph never receives provenance edges or gold spans.
- `provenance_path` ranks source-backed argument/output content units and may traverse the query-independent provenance graph.
- The provenance graph may change retrieval order and the diagnostic retrieved subgraph, but it never creates or expands gold.

All candidates retain exact `(event_id, json_pointer, char_start, char_end)` coordinates. Evaluation unions these coordinates rather than comparing method-local candidate IDs.

## Commands

The dataset config points to:

```text
data/isetrace/query-authoring/test-minimal-pilot-v7-3000.jsonl
```

Smoke runs:

```bash
uv run python experiment/run.py name=isetrace_v7_bm25_smoke dataset=isetrace profile=smoke method=bm25 device=cpu
uv run python experiment/run.py name=isetrace_v7_dense_smoke dataset=isetrace profile=smoke method=dense device=cuda:0
uv run python experiment/run.py name=isetrace_v7_graphrag_smoke dataset=isetrace profile=smoke method=graphrag device=cuda:0
uv run python experiment/run.py name=isetrace_v7_path_smoke dataset=isetrace profile=smoke method=provenance_path device=cuda:0
```

Full non-training runs should keep all method parameters frozen:

```bash
uv run python experiment/run.py -m \
  name=isetrace_v7_nontrain dataset=isetrace profile=full device=cuda:0 \
  method=bm25,dense,graphrag,provenance_path
```

## Interpretation

Report span Coverage/Recall, Span F1, Full Support, Evidence Density, MRR, and fixed token-budget results. Graph connectivity can remain a method diagnostic, but it is not a labeled dependency metric in v7.

The configured file name ends in `v7-3000`, but experiment scale is determined from parsed records rather than the filename. The current artifact contains 100 accepted query records over 50 authoring targets and 10 trajectories, so results from it must be described as a 100-query pilot, not a 3000-query run.

The generated query file is an authoring candidate set until separately reviewed and frozen. Review state is not embedded in the four-field records.

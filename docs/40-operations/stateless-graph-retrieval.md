# Stateless graph retrieval methods

This operation surface covers only graphrag and execution_provenance_retriever. It does not regenerate or migrate any dataset.

## GraphRAG

The method configuration owns seed_top_s, max_entity_document_frequency_ratio, sentence_resolver, min_sentence_score_margin, min_bridge_confidence, max_partners_per_anchor, and preserve_dense_top_n.

GraphRAG ranks all candidates with Dense first. It then derives typed title/body entity evidence, resolves each title group to at most one sentence with the same frozen encoder instance, and applies only local stable insertion. If no bridge moves a partner, the result is exactly Dense.

## Execution provenance (EPGM, non-trained)

There is exactly one non-trained EPGM implementation. `variant` selects a
frozen preset; there is no second method id and no second code path.

| variant | Traversal | Edge scope | Path score | Gating | Fusion | Role |
| --- | --- | --- | --- | --- | --- | --- |
| `typed_beam` (default) | bidirectional | all typed edges except `contains` | type prior x reverse factor x hop decay | none | additive | reported method |
| `dependency_path` | directed | `returns`/`feeds`/`grounds`/`supports`/`depends_on` | geometric mean of `feeds` weights x hop penalty | schema | stable insert | ablation of the default |

Both presets consume the existing `ExecutionProvenanceRankingRequest` graph and
reuse recorded semantic edge weights; neither rebuilds dataset graphs.

Under `typed_beam`, dense relevance is a floor that graph propagation can lift
but never demote, and audit edges (`supports`, `contradicts`, `invalidates`,
`verifies`) are traversable.

Under `dependency_path`, binding, path completeness (exactly one `feeds` plus
one `returns`), and lifecycle are hard validity gates, and insertion preserves
the dense score multiset exactly. Because the gate requires a
`feeds`/`returns` backbone, audit-only structure is always rejected as
`incomplete_path`, so on traces without that backbone the preset degenerates to
Dense. That degeneration is the reason `typed_beam` is the default.

Invalidated and superseded nodes stay reachable in both presets; lifecycle is
reported in the trace, never silently pruned.

Select a preset with either method YAML:

```
method=execution_provenance_retriever                    # typed_beam
method=execution_provenance_retriever_dependency_path    # ablation
method=execution_provenance_retriever method.variant=dependency_path
```

The active preset is recorded as `native_trace.variant` and as the run's
`graph_memory.variant` tag, and it participates in the Prefect cache key, so
the two presets never share cached rankings.

## Compatibility boundary

- Existing twowiki_provenance schema, converter, fixtures, prepared artifacts, and caches remain unchanged.
- Trainable R-GCN methods and checkpoints remain unchanged.
- Shared evaluation metrics and artifact roles remain unchanged.
- New native traces are method-local: typed_local_bridge and execution_provenance_local.

Use the normal experiment command with the existing method YAML. No data preparation command specific to this change is required.

The standalone real-trace runner takes the same preset flag:

```
uv run python scripts/run_epgm_provenance.py --epgm-variant typed_beam
```

It writes one JSONL per method, suffixed with the preset for EPGM rows.

# Stateless graph retrieval methods

This operation surface covers only graphrag and execution_provenance_retriever. It does not regenerate or migrate any dataset.

## GraphRAG

The method configuration owns seed_top_s, max_entity_document_frequency_ratio, sentence_resolver, min_sentence_score_margin, min_bridge_confidence, max_partners_per_anchor, and preserve_dense_top_n.

GraphRAG ranks all candidates with Dense first. It then derives typed title/body entity evidence, resolves each title group to at most one sentence with the same frozen encoder instance, and applies only local stable insertion. If no bridge moves a partner, the result is exactly Dense.

## Execution provenance

The method configuration owns seed_top_s, beam_width, max_hops, max_paths_per_seed, max_path_expansions, min_path_confidence, preserve_dense_top_n, and hop_penalty.

The method consumes the existing ExecutionProvenanceRankingRequest graph. Binding, path completeness, and lifecycle are validity gates. Existing semantic edge weights are consumed once; the method does not rebuild dataset graphs. If no path moves a partner, the result is exactly Dense.

## Compatibility boundary

- Existing twowiki_provenance schema, converter, fixtures, prepared artifacts, and caches remain unchanged.
- Trainable R-GCN methods and checkpoints remain unchanged.
- Shared evaluation metrics and artifact roles remain unchanged.
- New native traces are method-local: typed_local_bridge and execution_provenance_local.

Use the normal experiment command with the existing method YAML. No data preparation command specific to this change is required.

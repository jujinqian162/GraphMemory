# Stateless graph retrieval methods

This operation surface covers only graphrag and execution_provenance_retriever. It does not regenerate or migrate any dataset.

## GraphRAG

The method configuration owns seed_top_s, max_entity_document_frequency_ratio, sentence_resolver, min_sentence_score_margin, min_bridge_confidence, max_partners_per_anchor, and preserve_dense_top_n.

GraphRAG ranks all candidates with Dense first. It then derives typed title/body entity evidence, resolves each title group to at most one sentence with the same frozen encoder instance, and applies only local stable insertion. If no bridge moves a partner, the result is exactly Dense.

## Execution provenance (EPGM, non-trained)

There is one public non-trained EPGM implementation and one reported default:
`ppr_steiner`. Historical path strategies remain explicit diagnostics under the
same registry id.

| variant | Candidate generation | Graph objective | Recorded weights | Role |
| --- | --- | --- | --- | --- |
| `ppr_steiner` (default) | query-conditioned typed PPR over the full native graph | budgeted connected evidence subgraph | raw source-local transition factor | reported method; quality experiments pending |
| `typed_beam` | Dense top-seed bounded paths | independent additive target rerank | disabled by default | historical diagnostic |
| `dependency_path` | directed dependency paths | schema gate + stable insert | `feeds` geometric confidence | historical diagnostic |

### Default algorithm

1. The frozen Dense encoder ranks request candidates.
2. The same encoder compares the query with versioned, schema-owned natural-language descriptions of every public provenance relation.
3. Stored edges become forward and penalized reverse arcs. Arc strength combines relation affinity, a fixed type prior, native direction, degree control, and the raw recorded weight.
4. Arc strengths are normalized separately for each source node. RQ2's calibrated `feeds` weights therefore remain sibling-branch probabilities; RQ3's constant `1.0` values are an identity factor. There is no global min-max weight transform and no dataset-name branch.
5. Personalized PageRank diffuses a strictly positive min-max-scaled Dense teleport distribution over candidates and connector nodes; this preserves narrow cosine-score differences instead of flattening them with a unit-temperature softmax.
6. A deterministic budgeted Steiner-style greedy selector uses Dense/PPR prize minus a fixed candidate-inclusion cost and chooses a positive-marginal connected subgraph with at most `top_k` request candidates. Arc cost is `-log(transition_probability)` plus a fixed connector-hop cost, while arcs already in the selected tree have zero residual cost. Tool calls, agents, tasks, and other non-candidates may connect evidence without consuming the evidence budget.
7. An out-of-budget graph candidate must additionally pay the prize of the weakest replaceable Dense top-`k` incumbent. Accepted membership changes retain Dense-relative order and occupy the original descending Dense score slots, preventing graph-central evidence from destroying stronger early semantic hits.
8. Selected candidate paths are collapsed to oriented logical `feeds` edges for shared evaluation. The native trace preserves actual relation types, direction, transition probabilities, PPR mass, connectors, prizes, new-edge and displacement costs, marginal gains, and objective.

If no positive-marginal multi-candidate connection is selected, the full ranking and scores are exactly Dense. Structural session paths may affect ranking while remaining native-trace-only; only contracted evidentiary relations are emitted as shared logical dependencies.

### Running

```powershell
# Reported default
uv run python experiment/run.py `
  name=twowiki_provenance_epgm_ppr_steiner dataset=twowiki_provenance `
  profile=provenance_full device=cuda:0 method=execution_provenance_retriever

# Historical diagnostics only
uv run python experiment/run.py `
  name=twowiki_provenance_epgm_typed_beam dataset=twowiki_provenance `
  profile=provenance_full device=cuda:0 method=execution_provenance_retriever `
  method.variant=typed_beam

uv run python experiment/run.py `
  name=twowiki_provenance_epgm_dependency_path dataset=twowiki_provenance `
  profile=provenance_full device=cuda:0 method=execution_provenance_retriever `
  method.variant=dependency_path
```

The variant and a digest of every frozen behavior parameter participate in the
Prefect ranking implementation identity. Runs also record
`graph_memory.variant`, so caches and MLflow rows cannot silently cross
architectures.

The standalone real-trace runner uses the same default:

```bash
uv run python scripts/run_epgm_provenance.py \
  --epgm-variant ppr_steiner \
  --device cuda:0 \
  --output-dir runs/epgm_rq3
```

## Compatibility boundary

- Existing twowiki_provenance schema, converter, fixtures, and prepared artifacts remain unchanged.
- Trainable R-GCN methods and checkpoints remain unchanged.
- Shared evaluation metrics and artifact roles remain unchanged.
- Connector-only native nodes never enter the ranked candidate list or shared `retrieved_subgraph.nodes`.
- Quality claims for `ppr_steiner` remain pending server RQ2/RQ3 execution; unit tests establish invariants, not benchmark superiority.

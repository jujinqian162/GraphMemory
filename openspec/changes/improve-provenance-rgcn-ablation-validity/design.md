## Context

The current converter chooses two semantic successors per output and replaces the second successor with the gold target when the gold dependency is outside semantic top-2. It then exposes the original semantic rank through an artifact weight of `1 / rank`. A live read-only audit of the current converter over 629 accepted source-dev examples found that 77.6% of gold edges required this replacement, the median gold rank was 6, and the 90th percentile was 18. The visible graph therefore gives most gold dependencies much less message mass than ordinary rank-1/rank-2 branches.

The provenance R-GCN multiplies transformed messages by the artifact weight, while `wo_edge_weight` changes every edge to `1.0`. That control removes relative confidence and also increases total feed-message mass. Candidate and logical-edge heads are trained separately; inference ranks candidates before consulting the edge head, emits one successor per selected source without abstention, and training selects the checkpoint only by dev Full Support@5 over 500 examples. Generic BM25/dense hard negatives increase candidate-pair count from 444,072 to 1,036,168 in the delivered full run without a stable node-metric gain.

The intended benchmark remains label-conditioned by design: an ordered gold dependency is materialized first and hidden among ordinary typed branches. Labels may determine which subgraph must exist, but ranking input must not contain a gold flag or a visible convention that uniquely identifies that subgraph. The benchmark remains a controlled synthetic graph-retrieval task, not a claim that 2Wiki contains real agent executions.

This change crosses converter schema, pair construction, checkpoint schema, inference, evaluation, and experiment protocol. Existing v2 generated data and checkpoints are not compatible scientific inputs for the new ablation matrix.

## Goals / Non-Goals

**Goals:**

- Construct a deterministic gold-spine-plus-matched-branches graph in which the gold path is guaranteed but not uniquely identifiable by degree, relation, rank bucket, weight bucket, metadata shape, or ordering.
- Make semantic edge confidence bounded, source-local, and message-mass preserving so the full/weightless comparison isolates confidence allocation.
- Train candidate ranking against unique provenance-relevant competitors with a task-balanced ranking objective.
- Allow high-confidence learned logical edges to complete candidate sets through a bounded, auditable reranker and allow low-confidence edge predictions to abstain.
- Select checkpoints and report results with node and edge objectives that match inference behavior.
- Make every full/ablation comparison use identical dataset, split, encoder, construction, and evaluation identities and report uncertainty across seeds.

**Non-Goals:**

- Guaranteeing that the full model beats every ablation on every metric or using test results to tune construction/model thresholds.
- Changing standard `twowiki`, adding answer generation, or presenting the synthetic graph as an authentic agent-execution dataset.
- Jointly fine-tuning the dense encoder with the R-GCN, adding a beam decoder, or increasing R-GCN depth as part of this change.
- Replacing the generic evidence-graph R-GCN stack or changing non-provenance training-pair semantics.
- Preserving v2 raw-data or checkpoint compatibility.

## Decisions

### 1. Introduce schema-v3 gold-spine-plus-matched-branches construction

For each recoverable ordered pair `(gold_source, gold_target)`, the converter materializes the gold logical dependency before adding distractors. Candidate selection continues to guarantee both gold outputs and fill the remaining candidate cap with query-hard non-gold sentences. Candidate IDs, candidate order, and serialized edge order remain deterministically shuffled.

Every output receives exactly two `feeds` successors when the candidate pool permits:

1. a semantic head at complete-ranking position 1; and
2. a branch proposal selected from one of the versioned rank buckets `near=[2,4]`, `mid=[5,8]`, or `tail=[9,N]` using a stable hash of dataset seed, task ID, and source ID.

For the gold source, the gold target occupies the head slot when it ranks first and otherwise occupies the branch slot. If this displaces the source's hashed branch proposal, the converter assigns at least one non-gold source a branch from the same rank bucket as the gold edge. A record that cannot provide a non-gold bucket match while retaining fixed out-degree is rejected with a typed reason. This keeps the gold edge guaranteed without leaving rank>2 edges unique to the gold source.

All emitted branches still come from a complete source-aware semantic ranking; the branch policy chooses a deterministic rank-banded semantic proposal rather than an arbitrary endpoint. The manifest records the policy name, buckets, seed, candidate cap, successor count, scorer identity, query-template version, and all model/prefix/digest inputs.

The default scorer becomes source-aware hybrid ranking over `question + source evidence`, using the pinned frozen dense encoder and BM25 rank percentiles. BM25-only and dense-only remain explicit construction interventions. Hybrid weight and semantic temperature are explicit manifest inputs; the initial defaults are `0.5` and `0.1` respectively.

Alternative: keep top-2 and only increase the gold fallback weight. Rejected because a task-local special weight would make the required edge easier to identify and would leave the non-gold topology distribution unchanged.

Alternative: sample completely random tail endpoints. Rejected because random topology weakens branch semantics and conflicts with the benchmark's explainable-distractor contract.

### 2. Calibrate feed weights with a positive floor and fixed source mass

For a source with selected feed edges `E_s`, the converter transforms source-local semantic scores into probabilities `p_e` using temperature-softmax after scorer-owned normalization. With `d = |E_s|` and configured floor `f=0.5`, each artifact feed weight is:

`w_e = f + (1 - f) * p_e`

Therefore every feed weight lies in `[0.5, 1.0]` and every fixed-degree-2 source has total feed weight `1.5`. Non-feed structural edges remain `1.0`. Raw semantic score, complete semantic rank, confidence, calibrated weight, scorer identity, and branch bucket remain present on every feed edge under the same metadata schema.

`wo_edge_weight` retains every endpoint, relation ID, branch bucket, and direction, but replaces each feed weight with the arithmetic mean of the artifact weights for that source. Non-feed edges remain unchanged. Forward and reverse messages use the same resolved edge value. This preserves total source feed mass while removing within-source confidence allocation.

Alternative: add a second provenance-only message channel. Rejected for the first version because a bounded floor plus mass-matched ablation fixes the identified confound without changing the shared R-GCN layer interface.

### 3. Give pair construction a provenance-native typed context

The generic `TrainPairBuildTask` remains unchanged for flat and EvidenceGraph methods. Provenance pair routing projects a separate typed task containing `TextRankingRequest`, `ExecutionProvenanceGraph`, and `EvidenceLabel`, and delegates to a provenance-owned sampler.

The sampler materializes:

- positives for every gold output;
- `hard_provenance_successor` candidates that compete with the gold target from a gold source;
- `hard_provenance_predecessor` candidates that form competing logical transitions into a gold target;
- dense and BM25 hard candidates; and
- easy random candidates.

A candidate appears at most once per task. When sources overlap, the retained sample type follows `provenance_successor > provenance_predecessor > dense > bm25 > easy`, while the summary records overlap counts and all original sources. Initial full-profile counts per positive are `2/1/1/1/2` in that same category order, capped by unique eligible candidates.

Candidate loss becomes a task-balanced pairwise logistic loss: for each task, every selected positive is compared with every unique selected negative; comparisons are averaged within task and then tasks are averaged. The existing class-balanced logical-edge loss remains and is logged separately. `wo_hard_negatives` removes provenance, dense, and BM25 negatives, retains easy random negatives, and continues to invalidate from the pair stage.

Alternative: online model-hard mining each epoch. Deferred because it would make the materialized pair artifact cease to be the full reproducible supervision contract.

### 4. Use logical-edge predictions through bounded stable promotion

Raw candidate logits produce the base complete ranking. Structured inference considers only the first `structured_pool_size=16` candidates and only sources within `structured_seed_top_s=5`. A logical transition is eligible when both endpoints are in the pool and its sigmoid probability is at least `edge_accept_threshold=0.5`.

Each seed keeps at most its highest-probability eligible successor. Conflicts over one target are resolved by edge probability, original source rank, original target rank, and target ID. An accepted target below its source is inserted directly after the source subject to `preserve_node_top_n=2`; protected-prefix members and targets already above their source are no-ops. All non-promoted candidates preserve base relative order. After a real promotion, the descending multiset of original candidate scores is assigned to the final order, and trace data records original/final ranks plus the responsible edge.

The new `wo_edge_rerank` variant uses the same trained checkpoint and edge predictions but skips promotion, returning the raw candidate order. Existing model-changing variants retrain and use the same structured inference contract. This separates the value of edge-aware set completion from graph encoding and edge-head training.

Logical-edge output uses the same threshold and may abstain. It emits at most one successor per selected source and only emits transitions whose endpoints are in final top-k. Rejected and no-op transitions remain diagnostic trace entries rather than predicted edges.

Alternative: add edge logits directly to every candidate score. Rejected because global fusion can reward high-degree/noisy components and makes the movement harder to audit.

### 5. Select checkpoints with a joint dev objective

Provenance training evaluates the complete available generated dev split and logs Full Support@5, MRR, Edge Precision@10, Edge Recall@10, Edge F1@10, average emitted edges, and abstention rate after the effective variant's inference policy.

The checkpoint score is declared as:

`dev_joint = 0.50 * FullSupport@5 + 0.25 * MRR + 0.25 * EdgeF1@10`

Ties are broken by higher Full Support@5, then higher Edge F1@10, then higher MRR, then earlier epoch. Candidate loss, edge loss, and joint metrics are stored in training metrics and checkpoint metadata. The provenance checkpoint family advances to schema version 3 and rejects v2 checkpoints rather than guessing the new weight, loss, or inference semantics.

Alternative: continue selecting only Full Support@5. Rejected because it does not observe edge-head quality even though the edge head now affects ranking and emitted structure.

### 6. Complete edge evaluation and separate correctness from scientific outcome

Evaluation computes micro Edge Precision@10, Edge Recall@10, and Edge F1@10 from predicted logical dependency edges whose endpoints are in top-10. Zero predicted edges produce precision and F1 of zero, not `N/A`. Average emitted edges and abstention rate remain guardrails against recall-through-overproduction.

The formal matrix uses the same v3 train/dev/test digests for `full_rgcn`, `wo_graph`, `wo_edge_type`, `wo_edge_weight`, `wo_hard_negatives`, and `wo_edge_rerank`, with seeds `13`, `17`, and `29`. Each seed gets the same hyperparameter budget and full dev split. Delivery reports per-seed rows, mean and standard deviation, query-paired confidence intervals for deltas, and exact artifact/config identities.

Correctness gates are contract/invariant tests, not a desired metric ordering. A component contribution is claimed only when its paired interval and mechanism metrics support the claim. A saturated Full Support reversal of a few examples is reported as uncertainty rather than treated as an implementation failure.

Two diagnostic controls complement the public suite:

- topology-free candidate inference (`wo_graph`), already public; and
- deterministic shuffled-feed message endpoints, preserving relation/weight histograms while changing message topology and never replacing the real request graph used for labels or evaluation.

The shuffled control is diagnostic-only and owns a distinct model/cache identity. It is not silently substituted for `wo_graph` or included in the default ablation table.

## Risks / Trade-offs

- **[Hybrid construction raises conversion cost]** → Batch frozen-dense encoding, pin encoder identity in the manifest, keep conversion one-time, and retain BM25-only audit mode for fast checks.
- **[Gold-conditioned bucket matching remains synthetic]** → State this explicitly, keep all task-local correctness fields label-side, and require matched non-gold branches plus shuffled/topology-free controls before graph-reasoning claims.
- **[Rank-banded branches may be too weak or too noisy]** → Publish per-split rank/score/weight distributions and rejection counts; change defaults only through a new schema/config identity and dev-only evidence.
- **[Structured promotion can propagate a wrong edge]** → Bound pool/seeds, protect the prefix, threshold edge probability, permit abstention, and record every movement.
- **[Joint checkpoint weights are a new modeling choice]** → Declare them in config/checkpoints, report every component, and apply the same selection rule/budget to all variants.
- **[New metrics alter table schemas]** → Version evaluation outputs and update table validators/collectors atomically; do not reinterpret old result rows as v3.
- **[Three full seeds are expensive]** → Require focused unit/smoke gates before server runs and execute the full model plus one ablation family at a time without reducing the paired protocol.

## Migration Plan

1. Add schema-v3 records, scorer/branch configuration, statistics, validation, and converter tests without changing the active dataset config.
2. Generate pilot v3 raw data under `data/twowiki_provenance/v3/raw`, verify deterministic hashes and leakage/weight/bucket audits, then point `configs/dataset/twowiki_provenance.yaml` at the v3 directory.
3. Add provenance pair routing, sample types, deduplication, summaries, and task-balanced candidate loss; invalidate pair artifacts through implementation/config identity.
4. Add calibrated weight policies, structured inference, abstention, new variant, joint dev evaluation, and checkpoint schema v3.
5. Add evaluation columns, validators, delivery surfaces, runbook commands, and focused behavioral tests.
6. Run one-seed quick experiments to validate runtime and artifact equality, then run the three-seed full matrix and publish uncertainty-aware ablation tables.
7. Roll back by restoring the dataset config to v2 raw data and the previous public method config; v2 and v3 artifacts remain isolated by schema/digest and no compatibility shim is retained.

## Open Questions

None blocking. Initial numeric defaults are explicit experiment inputs and may be changed only before the formal test matrix using train/dev construction audits; any post-test change requires a new construction/model identity and a fresh matrix.

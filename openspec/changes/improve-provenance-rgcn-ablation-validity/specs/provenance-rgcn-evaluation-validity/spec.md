## ADDED Requirements

### Requirement: Provenance evaluation reports edge precision and F1
For methods that emit logical dependency edges, evaluation SHALL compute micro Edge Precision@10, Edge Recall@10, and Edge F1@10 from predicted edges whose endpoints are in the candidate top-10. It SHALL also report average emitted logical edges and source abstention rate.

#### Scenario: Predictions contain true and false logical edges
- **WHEN** evaluation aggregates predicted logical edges across tasks
- **THEN** precision uses total true positives divided by total predicted edges, recall uses total true positives divided by total gold edges, and F1 is their harmonic mean

#### Scenario: Method emits no logical edges
- **WHEN** no edge is predicted for the evaluated population
- **THEN** Edge Precision@10 and Edge F1@10 are `0.0`, recall is computed from missed gold edges, and the metrics are not reported as `N/A`

### Requirement: Checkpoint selection observes node and edge quality
Every provenance R-GCN epoch SHALL evaluate the complete configured dev population and compute `dev_joint = 0.50 * Full Support@5 + 0.25 * MRR + 0.25 * Edge F1@10` under the effective variant's inference policy. The checkpoint with highest joint score SHALL be retained with deterministic tie-breaking by Full Support@5, Edge F1@10, MRR, and earlier epoch.

#### Scenario: Node coverage improves while edge quality collapses
- **WHEN** a later epoch raises Full Support@5 but lowers the declared joint score because Edge F1 or MRR degrades
- **THEN** the later epoch does not replace the current best checkpoint

#### Scenario: Two epochs have the same joint score
- **WHEN** joint scores are equal at stored precision
- **THEN** the declared component ordering and earlier-epoch rule choose one checkpoint deterministically

### Requirement: Formal ablations use the full available dev split
The formal v3 provenance R-GCN run configuration SHALL prepare all available generated dev records for checkpoint selection and SHALL keep the generated test split untouched until all construction, model, threshold, and checkpoint rules are frozen.

#### Scenario: Full-profile formal run is composed
- **WHEN** the provenance R-GCN ablation matrix is planned
- **THEN** its resolved dev split count equals the available v3 dev count rather than the historical fixed 500-example subset

#### Scenario: Test results motivate a setting change
- **WHEN** an operator proposes changing construction, thresholds, loss weights, or selection weights after reading test metrics
- **THEN** the change receives a new construction/model identity and the previous test split is not reused as confirmation evidence

### Requirement: Compared variants have identical scientific inputs
Within one seed, full and ablation runs SHALL have identical raw, prepared train/dev/test, encoder, construction, candidate, label, evaluation, and non-ablated configuration digests. Only dimensions declared by the selected variant and their required downstream artifacts MAY differ.

#### Scenario: Model-only ablation is compared
- **WHEN** `wo_graph`, `wo_edge_type`, or `wo_edge_weight` is evaluated
- **THEN** pair and dataset digests equal full while model, prediction, and evaluation identities record the variant

#### Scenario: Pair-changing ablation is compared
- **WHEN** `wo_hard_negatives` is evaluated
- **THEN** dataset digests equal full, pair digest differs for the declared sampling change, and every downstream origin references that pair digest

#### Scenario: Ranking-only ablation is compared
- **WHEN** `wo_edge_rerank` is evaluated
- **THEN** dataset, pair, and model digests equal full while prediction and evaluation identities differ

### Requirement: Formal evidence uses paired multi-seed reporting
The formal ablation matrix SHALL run seeds `13`, `17`, and `29` with the same hyperparameter budget. Delivery SHALL include every seed row, mean, standard deviation, query-paired confidence intervals for full-minus-ablation deltas, and discordant-example counts for Full Support metrics.

#### Scenario: One seed reverses a saturated metric
- **WHEN** an ablation is slightly higher on Full Support@10 for one seed but the paired interval spans zero
- **THEN** delivery reports the reversal and uncertainty without declaring the ablation superior or the implementation invalid

#### Scenario: Component contribution is claimed
- **WHEN** a report states that one full-model component improves an outcome
- **THEN** the claim identifies the primary metric, mean paired delta, uncertainty interval, and supporting mechanism metric

### Requirement: Correctness does not encode a desired leaderboard ordering
Automated acceptance SHALL validate schemas, determinism, leakage constraints, message-mass equality, lifecycle invalidation, metric arithmetic, trace behavior, and artifact provenance. It MUST NOT fail solely because a full-model test metric is equal to or lower than an ablation metric.

#### Scenario: Ablation wins a valid metric
- **WHEN** all contract checks pass but an ablation has a higher measured test score
- **THEN** the experiment remains valid and the result is reported as evidence about that component

#### Scenario: Full model wins with unfair inputs
- **WHEN** full and ablation rows use different undeclared dataset, dev, encoder, or evaluation identities
- **THEN** the comparison fails provenance validation regardless of the metric ordering

### Requirement: Shuffled-feed topology is an isolated diagnostic control
The system SHALL provide a deterministic diagnostic model control that permutes feed-message endpoints within each task while preserving relation and weight histograms. It SHALL retain the original request graph for label/evaluation semantics, own a distinct cache identity, and remain outside the default public ablation table.

#### Scenario: Shuffled control is repeated
- **WHEN** the same request digest and shuffle seed are used
- **THEN** the tensorized feed-message permutation is identical and non-feed messages are unchanged

#### Scenario: Default ablation matrix is planned
- **WHEN** an operator runs the public provenance suite without explicitly requesting diagnostics
- **THEN** the shuffled-feed control is absent from the aggregate table

### Requirement: V3 evaluation and delivery schemas are explicit
Metric contracts, tables, validators, run summaries, MLflow metric names, and collected result deliveries SHALL include the new edge metrics and effective experiment identities under a versioned v3 evaluation schema. Old rows SHALL remain readable as historical artifacts but MUST NOT be merged into a v3 ablation aggregate.

#### Scenario: V3 row is aggregated
- **WHEN** a provenance-capable method emits a v3 evaluation row
- **THEN** the row contains Edge Precision@10, Edge Recall@10, Edge F1@10, average emitted edges, abstention rate, variant, seed, and input identities

#### Scenario: V2 and v3 rows are selected together
- **WHEN** aggregation receives incompatible evaluation schema versions
- **THEN** it fails with an explicit schema mismatch instead of filling missing columns or comparing the rows

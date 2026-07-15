## 1. Joint Action Calibration and Score Normalization

- [x] 1.1 Add failing loss tests proving that shifting every candidate logit changes the candidate-versus-STOP probability, completed states supervise STOP against distractors, and unselected future-gold nodes remain masked from both the numerator and denominator.
- [x] 1.2 Implement a shared candidate-plus-STOP action distribution and update next-action and stop-loss calculation so the retained stop component is derived from the joint STOP probability.
- [x] 1.3 Add failing decoder tests for action-count normalization: each candidate expansion adds one action, STOP adds one action, and stopping from the empty state has action count one.
- [x] 1.4 Move score normalization and beam pruning behind the decoder-owned helper used by both training and inference, then remove the duplicate training-side normalization path.

## 2. Last-Node-Aware Beam Deduplication

- [x] 2.1 Add failing decoder and training tests showing that states with the same selected set and same last node collapse, while states with the same selected set but different last nodes remain distinct.
- [x] 2.2 Change the shared deduplication identity to `(selected_set, last_node)`, using `None` as the last node for the empty state.
- [x] 2.3 Add failing training and inference tests showing that `deduplicate_selected_sets: false` preserves otherwise duplicate states.
- [x] 2.4 Propagate `deduplicate_selected_sets` into training beam construction and make the shared pruning helper bypass identity-based filtering when it is disabled.

## 3. Config-Authoritative R-GCN Checkpoint Selection

- [x] 3.1 Add configuration tests for every supported R-GCN selection metric, rejection of unknown metrics, and the canonical `dev_composite` selection setting.
- [x] 3.2 Introduce a model-domain selection settings record and metric resolver covering `dev_composite`, `dev_full_support_at_5`, `dev_full_support_at_10`, `dev_recall_at_5`, `dev_mrr`, and `dev_loss`, including each metric's optimization direction.
- [x] 3.3 Convert and pass `settings.selection` from `RgcnGraphRetrieverTrainer` into `train_graph_retriever`, replacing the hard-coded best-metric choice while retaining a direct-call default equivalent to the canonical configuration.
- [x] 3.4 Add checkpoint-selection tests for the composite metric, an alternative higher-is-better metric, and lower-is-better `dev_loss`.
- [x] 3.5 Record the effective selection metric name, current value, and best value in training metrics without changing the checkpoint tensor schema.
- [x] 3.6 Update both canonical R-GCN YAML configurations to `best_metric: dev_composite` and `higher_is_better: true`; add an adjacent comment explaining that `trainer.learning_rate` is retained only for the current config shape and auditability because optimizer learning rates come from the phase-specific fields, and test that changing only this legacy field does not alter optimizer learning rates.

## 4. Complete Beam Development Loss

- [x] 4.1 Add failing development-evaluation tests proving that `dev_loss` is the configured weighted beam objective rather than base-scorer BCE and that its next-action, stop, and auxiliary components are emitted separately.
- [x] 4.2 Extract a model-domain beam-loss helper shared by training and development evaluation, updating existing callers and tests so evaluation does not import a training-lifecycle implementation detail.
- [x] 4.3 Compute the full development beam objective under evaluation mode and no-grad, using all development candidates, the configured loss weights, the training beam policy, and the effective training auxiliary positive-class weight.
- [x] 4.4 Wire `dev_loss`, `dev_next_action_loss`, `dev_stop_loss`, and `dev_aux_node_loss` into epoch metrics and expose `dev_loss` to the configurable selection resolver.
- [x] 4.5 Verify with a regression test that an existing beam R-GCN checkpoint still loads after the metric and loss-accounting changes.

## 5. Verification

- [x] 5.1 Run the focused decoder, beam-loss, deduplication, configuration, selection, metric-recording, and checkpoint tests introduced by this change.
- [x] 5.2 Run the complete R-GCN configuration, model, tensorization, training, retrieval, checkpoint, and tracking test suites.
- [x] 5.3 Run formatting, linting, static type checking, compilation checks, and strict OpenSpec validation for the completed change.
- [x] 5.4 Run fresh smoke workflows for both existing R-GCN public methods and verify the effective selection configuration, development loss components, prediction metadata, and checkpoint output; do not add an ablation or historical comparison run to this change.

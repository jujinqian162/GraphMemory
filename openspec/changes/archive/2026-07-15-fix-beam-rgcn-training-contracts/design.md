## Context

The completed `add-beam-rgcn-evidence-decoder` change introduced selection-conditioned decoding for the two existing R-GCN methods. Review of the live implementation found four contract gaps that matter before full training: training normalizes candidates separately from the raw `STOP` logit while search normalizes them together; stopped hypotheses add a `STOP` log-probability without adding an action to the length denominator; pruning treats all sequences with the same selected set as equivalent even though the decoder consumes the last selected node; and the stage-level selection config is not passed into R-GCN training.

The current `dev_loss` is also only the base node-scorer BCE. It is therefore not comparable to the beam-aware training objective and cannot diagnose the decoder loss components. The canonical R-GCN configs already contain a deduplication switch, a selection block, and both legacy and optimizer-phase learning-rate fields, so this change should make those live contracts authoritative rather than add another configuration layer.

## Goals / Non-Goals

**Goals:**

- Calibrate candidate and `STOP` actions in the same supervised probability space used by beam search.
- Use one action-count normalization rule in training and inference.
- Preserve distinct future states by deduplicating on `(selected_set, last_node)`.
- Honor `deduplicate_selected_sets` in training as well as inference.
- Make typed R-GCN checkpoint selection config-authoritative while retaining the current composite default behavior.
- Make `dev_loss` the full weighted beam objective and expose its components.
- Explain in the canonical YAML why `trainer.learning_rate` is retained but does not control either optimizer parameter group.
- Preserve the existing public method ids, model architecture, checkpoint tensor state, ranked-result contract, and workflow topology.

**Non-Goals:**

- Change `wo_graph`, `wo_edge_type`, frontier, or other ablation semantics.
- Add a legacy one-shot comparison method or formal MLflow comparison workflow.
- Decouple configured training and inference beam widths.
- Vectorize Python-side beam expansion or otherwise optimize beam throughput.
- Change exported ranked-node score semantics.
- Tune beam, loss, optimizer, depth, or decoder-capacity hyperparameters.

## Decisions

### 1. Supervise candidates and `STOP` in one action distribution

For each non-stopped hypothesis, the supervised action distribution contains the currently eligible unselected candidates plus `STOP`. Gold candidates masked by the dynamic oracle because their predecessors are incomplete remain excluded from the supervised numerator and denominator, preserving the existing future-gold masking contract.

The next-action loss is the negative log probability mass assigned to all valid actions. Before gold completion, the valid actions are the dependency-ready gold candidates and `STOP` remains in the denominator as an invalid action. After completion, `STOP` is the sole valid action and all remaining distractors stay in the denominator. This directly calibrates the relative offset of candidate and stop scores.

The existing `stop_loss_weight` remains meaningful: the stop component is computed from the `STOP` probability in this joint distribution rather than from an independently interpreted raw stop logit. It is an auxiliary binary stop/completion signal, while the joint next-action loss remains authoritative for beam action calibration. The auxiliary node BCE remains unchanged except where development evaluation uses the full candidate set.

**Alternative considered:** remove the separate stop component. This is simpler, but it would make an existing typed loss setting and diagnostic disappear. Deriving it from the joint probability fixes calibration without changing the public loss configuration shape.

### 2. Normalize by the number of actions actually scored

An active hypothesis with `k` selected candidates has taken `k` actions. A stopped hypothesis with the same selected prefix has taken `k + 1` actions because `STOP` is an action whose log-probability is included in the raw sequence score. Empty-prefix `STOP` therefore has action count one.

The decoder module will own the shared score-normalization and pruning helpers used by both inference and beam-aware training. The duplicate training-only normalization implementation will be removed so the two paths cannot drift again.

### 3. Define hypothesis identity as `(selected_set, last_node)`

The decoder's subsequent-hop state contains both a permutation-insensitive selected-set context and a last-node context. Hypotheses are therefore equivalent for future scoring only when both their selected set and last selected node match. The empty hypothesis uses `last_node=None`.

When `deduplicate_selected_sets=true`, pruning retains the highest-scoring hypothesis for each `(frozenset(selected_indices), last_selected_index)` key. Different orders that end at different last nodes remain eligible for separate beam slots. When the setting is false, neither training nor inference performs state-key deduplication; both only apply deterministic score ordering and beam-width truncation.

**Alternative considered:** remove last-node conditioning and retain set-only identity. The user selected last-node-aware identity, and removing the signal would change decoder capacity beyond this corrective change.

### 4. Make selection config authoritative without changing the default result

`RgcnGraphRetrieverTrainer` will convert `settings.selection` into a model-domain selection settings record and pass it to `train_graph_retriever`. Training will resolve the selected value from an explicit metric map and compare it according to `higher_is_better`.

Supported R-GCN selection names are:

- `dev_composite`, defined as `0.50 * Full Support@5 + 0.30 * Recall@5 + 0.20 * MRR`;
- `dev_full_support_at_5`;
- `dev_full_support_at_10`;
- `dev_recall_at_5`;
- `dev_mrr`;
- `dev_loss`.

Unknown names fail typed validation before execution. Both canonical R-GCN YAML files change from `full_support_at_5` to `dev_composite` with `higher_is_better: true`. This makes the config truthful while preserving the implementation's current effective checkpoint choice. Direct model-domain training calls default to the same composite selection unless they pass explicit settings.

Selection settings do not change the checkpoint tensor schema in this change. The stage's effective config and training metrics record the selected metric name and value, while the existing checkpoint continues to record the chosen `best_dev_metric` and epoch.

### 5. Compute development loss from the complete beam objective

Beam-loss computation will be extracted into a shared model-domain helper that does not import the training lifecycle. Training and development evaluation will use the same joint next-action loss, joint-probability stop loss, auxiliary-node BCE, configured weights, beam-state policy, and effective positive weighting.

Development batches cover every candidate rather than sampled training pairs. Consequently, the development auxiliary term is intentionally computed over all development candidates using the effective training `pos_weight` when enabled. `dev_loss` is the resulting weighted total, and metric records also expose `dev_next_action_loss`, `dev_stop_loss`, and `dev_aux_node_loss`.

Development loss is computed in evaluation mode without gradients. Existing evidence metrics and checkpoint-selection candidates remain calculated from beam-decoded rankings.

### 6. Retain and annotate the legacy trainer learning rate

`trainer.learning_rate` remains in the typed config and checkpoint training record to avoid an unrelated contract migration. Both canonical R-GCN YAML files receive an inline comment stating that the field is retained for audit/current shape only and that optimizer execution uses `optimizer_phases.decoder_learning_rate` and `optimizer_phases.rgcn_learning_rate`.

No optimizer code will read `trainer.learning_rate`, and no aliasing or synchronization rule will be introduced.

## Risks / Trade-offs

- **[Joint stop supervision changes optimization dynamics]** → Preserve existing loss weights, add focused probability-shift tests, and keep the current composite checkpoint-selection default.
- **[Last-node-aware identity uses more beam slots than set-only identity]** → Keep the configured beam bound and deterministic ordering; this is the intended cost of retaining non-equivalent future states.
- **[Full beam dev loss adds evaluation work]** → Reuse the shared loss implementation and existing dev batches; throughput vectorization remains explicitly out of scope.
- **[A configured lower-is-better metric is compared incorrectly]** → Test both `higher_is_better=true` and `false`, including `dev_loss` selection.
- **[Metric-name drift reintroduces silent fallback]** → Use a closed typed set of supported metric names and reject unknown values before planning or execution.

## Migration Plan

1. Add failing tests for joint action calibration, stopped action counts, and last-node-aware state identity.
2. Share beam normalization/pruning between training and inference and honor the deduplication setting in both paths.
3. Extract the beam-loss calculation and switch training and dev evaluation to the calibrated joint action distribution.
4. Add typed R-GCN selection settings, pass them through the stage trainer, and make checkpoint selection metric/direction configurable.
5. Update both canonical R-GCN configs to `dev_composite` and annotate the legacy trainer learning-rate field.
6. Update metric/checkpoint tests, run focused and full R-GCN verification, and complete fresh smoke workflows for both R-GCN methods.

Rollback is a code/config revert. Existing beam checkpoints remain loadable because model tensors and checkpoint fields do not change, but continued training across the boundary uses the objective implemented by the checked-out revision.

## Open Questions

None. Ablation redesign, historical comparisons, beam throughput, ranked-score semantics, and hyperparameter selection remain intentionally outside this change.

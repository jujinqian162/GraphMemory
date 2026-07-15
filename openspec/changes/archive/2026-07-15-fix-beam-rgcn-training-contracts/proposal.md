## Why

The beam R-GCN workflow is runnable, but its current training and selection contracts do not exactly match inference: `STOP` is not calibrated in the same action distribution, stopped hypotheses use inconsistent length accounting, set-only deduplication discards distinct last-node states, and the configured checkpoint-selection metric is ignored. These issues must be corrected before server-side full training so that the existing defaults keep their intended behavior while alternative valid settings are actually honored.

## What Changes

- Train candidate actions and `STOP` in one calibrated action probability space, while retaining the configured stop-loss component as an auxiliary signal derived from that same probability.
- Count `STOP` as an action when normalizing a stopped hypothesis score, and share the same normalization semantics between training and inference.
- Deduplicate beam hypotheses by `(selected_set, last_node)` instead of `selected_set` alone.
- Make training honor `deduplicate_selected_sets`; disabling it must disable deduplication in both training and inference.
- Pass typed R-GCN selection settings from `RgcnGraphRetrieverTrainer` into the training lifecycle and select checkpoints by the configured metric and direction.
- Change both canonical R-GCN configs to `best_metric: dev_composite`, preserving the current effective composite-selection behavior after the configuration path becomes authoritative.
- Redefine `dev_loss` as the complete weighted beam objective and report its next-action, stop, and auxiliary-node components.
- Keep the legacy `trainer.learning_rate` field for the current config/checkpoint shape, but annotate both canonical YAML files that the optimizer uses `optimizer_phases.decoder_learning_rate` and `optimizer_phases.rgcn_learning_rate` instead.
- Add regression tests for joint `STOP` calibration, action-count normalization, last-node-aware deduplication, disabled deduplication, configurable selection, and complete beam dev loss.
- Do not change graph/frontier ablations, add historical comparison workflows, alter public method ids, or address unrelated beam performance and ranked-score semantics in this change.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `beam-rgcn-evidence-decoding`: Align beam training, normalization, hypothesis identity, deduplication, and development loss with the inference contract.
- `current-trainable-method-config`: Make R-GCN checkpoint selection authoritative from typed config, set the canonical default to the existing composite metric, honor the deduplication switch during training, and document the legacy trainer learning-rate field.

## Impact

- Affected model code: `graph_memory/models/graph_retriever/decoder.py`, `training.py`, `dev_evaluation.py`, and shared beam-loss/pruning helpers introduced or extracted during implementation.
- Affected workflow code: `graph_memory/stages/trainers.py` and the typed R-GCN selection handoff.
- Affected configuration: `configs/method_configs/dense_rgcn_graph_retriever.yaml` and `dense_ft_rgcn_graph_retriever.yaml`.
- Affected metrics: `dev_loss` changes from base-scorer BCE to the complete weighted beam loss; component metrics make the new meaning explicit.
- Affected verification: focused decoder/loss/selection tests, current R-GCN train/retrieve tests, strict OpenSpec validation, static checks, and fresh smoke consuming workflows for both R-GCN methods.

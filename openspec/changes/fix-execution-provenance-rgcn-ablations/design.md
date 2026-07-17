## Context

The experiment system already owns ablation discovery, validation, artifact namespaces, baseline aliases, invalidation boundaries, and aggregation through `graph_memory.registry.ablations` and `graph_memory.experiment.planning`. That implementation currently registers only the two evidence-graph R-GCN methods, and its patch helper narrows method configs to those two types. The execution-provenance R-GCN is a separate trainable method with typed relations, artifact edge weights, node types, candidate supervision, logical-edge supervision, and hard-negative pair sampling, but its public method config has no model ablation field.

The implementation must preserve the distinction between evidence graphs and execution-provenance graphs. Evidence-only variants such as `wo_bridge` have no faithful provenance equivalent and would be no-op experiments.

## Goals / Non-Goals

**Goals:**

- Make supported execution-provenance R-GCN variants selectable through the existing root `ablation` config.
- Reuse existing planner lifecycle semantics and output layout rather than adding a provenance-specific runner.
- Make each registered variant produce a concrete, checkpointed change in model or pair construction.
- Preserve ordinary execution-provenance R-GCN behavior and existing checkpoint loading.

**Non-Goals:**

- Do not register evidence-only variants for the provenance model.
- Do not add topology-only or shuffled-edge graph controls in this bug fix; those require a separate graph-transform contract rather than a model/pair patch.
- Do not change candidate or logical-edge loss definitions, checkpoint selection metrics, dataset construction, or public method IDs.
- Do not change the root default variant list, which remains the evidence-R-GCN default; provenance-only runs select a supported explicit subset.

## Decisions

### Decision 1: Register a separate four-variant provenance suite

The registry will add an `EXECUTION_PROVENANCE_RGCN_ABLATION_PATCH_SUITE` containing the shared `full_rgcn` baseline alias plus `wo_graph`, `wo_edge_type`, `wo_edge_weight`, and `wo_hard_negatives`. The first three invalidate from train; `wo_hard_negatives` invalidates from pairs.

Reusing the full evidence suite was rejected because bridge/entity/sequential/query-overlap edges and dense seed-score features do not exist in the provenance model. Registering them would create misleading no-op rows.

### Decision 2: Generalize the existing typed patch path

`ExecutionProvenanceRgcnMethodConfig` will gain the same required `train.model.ablation` field used by evidence R-GCN configs. `_variant_invocations` and `_apply_rgcn_ablation` will accept this method config in their typed unions. Existing `RgcnModelPatch`, `NoGraphModelPatch`, and `PairSamplingPatch` remain the registry-owned patch records, because their update targets are structurally shared and their invalidation semantics already match.

A separate provenance workflow branch was rejected because pair, train, retrieve, evaluate, alias, and aggregate construction are already method-polymorphic after the config patch is applied.

### Decision 3: Resolve an ablation name into explicit checkpointed model policies

The provenance model package will provide one default model-config constructor. It resolves `full_rgcn` to typed relation transforms plus artifact edge weights, `wo_graph` to zero layers, `wo_edge_type` to a shared relation transform, and `wo_edge_weight` to uniform weights. The resulting `ProvenanceRgcnModelConfig` records `ablation_name`, `message_transform_type`, and `edge_weight_policy`; the model and tensorizer consume the explicit policies rather than branching independently on a string.

The checkpoint loader will default missing policy fields to the full-model behavior so existing schema-version-2 checkpoints remain readable. Bumping the schema was rejected because the model parameter layout and old full-model meaning are unchanged.

### Decision 4: Keep hard-negative removal as a pair-stage patch

`wo_hard_negatives` continues to use `PairSamplingPatch`, setting all hard-negative counts to zero while leaving `easy_random_per_positive` unchanged. This produces variant-local pairs and correctly invalidates every downstream stage.

Moving negative sampling into the model config was rejected because sampling happens before training and is already owned by the pair stage.

### Decision 5: Verify the public Hydra surface, not only helper functions

Tests will cover registry inspection, resolved config composition, exact planner invocations/aliases/selections, patch values, model transform construction, tensor edge weights, and the disabled-ablation regression. A real `experiment/plan.py` command using the requested override syntax will be run as the final acceptance check.

## Risks / Trade-offs

- **[The global default list contains evidence-only variants]** → Document and test an explicit supported `ablation.variants` subset for provenance-only runs; do not silently ignore unsupported names.
- **[Old checkpoints lack the new explicit policy fields]** → Decode absent fields as `full_rgcn`, typed transforms, and artifact weights.
- **[A registered variant could accidentally become a no-op]** → Add behavior tests at the model/tensor boundary in addition to planner snapshots.
- **[Mixed-method experiments can select a variant supported by only some suites]** → Preserve current registry semantics: each variant expands only for selected methods that register it.

## Migration Plan

1. Add failing registry/planner/model tests.
2. Add the provenance suite and typed config field.
3. Generalize planner patching and propagate effective policies into training checkpoints.
4. Update inspection/runbook documentation and execute focused tests plus a real plan command.
5. Roll back by removing the suite registration and provenance-specific policy fields; ordinary runs remain unaffected throughout.

## Open Questions

None. Topology-only and shuffled-edge controls remain explicitly deferred to a graph-transform change.

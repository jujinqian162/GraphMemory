## Context

The current user entry point is `scripts/experiment.py`, but most orchestration behavior lives in `graph_memory/experiment.py`. That helper owns config resolution, manifest generation, artifact layout, stage selection, dependency validation, status inspection, and concrete low-level command construction.

The current runner works for the implemented methods, but lifecycle decisions are inferred from broad flags such as `requires_checkpoint` and from central special cases for graph rerank and trainable graph retrieval. `requires_checkpoint` currently implies the R-GCN-specific `pairs -> train -> retrieve -> evaluate` lifecycle. That assumption will not fit Dense-FT, Memory Stream, GraphRAG, or a future trainable graph method with different supervision artifacts.

R-GCN model code already supports structural variants such as `wo_graph`, `wo_edge_type`, `wo_bridge`, `wo_edge_weight`, and `wo_seed_score`. The repository also contains an ablation config draft, but the runner does not expand variants, allocate isolated outputs, reuse unchanged upstream artifacts, resume variant runs, or aggregate `ablation_results.csv`.

The design must preserve two stable repository boundaries:

- `scripts/experiment.py` remains the user-facing experiment entry point.
- Low-level scripts remain explicit IO adapters with inspectable input and output paths.

New orchestration modules will live under `scripts/workflow/` so `scripts/` does not become a flat collection of planning helpers.

## Goals / Non-Goals

**Goals:**

- Move experiment lifecycle knowledge out of central method-name branches and into small workflow adapters.
- Represent closed orchestration vocabularies with explicit enum types rather than unconstrained `str` values.
- Model ablation as a method-variant experiment matrix, not as an opaque lifecycle stage.
- Let each variant declare changed dimensions and each workflow step declare invalidating dimensions.
- Let the planner calculate the first invalidated step and alias reusable upstream artifacts without variant-name branches.
- Add config-controlled R-GCN ablation planning, execution, strict downstream resume, status inspection, and `tables/ablation_results.csv`.
- Preserve existing non-ablation commands, run directories, manifests, and low-level script contracts where practical.
- Make a later same-lifecycle retriever a registry addition and a later different-lifecycle retriever a local workflow-adapter addition.

**Non-Goals:**

- Do not introduce dynamic plugin discovery, arbitrary workflow loading from config, or a general DAG engine.
- Do not implement Dense-FT, Memory Stream, or GraphRAG in this change.
- Do not expand random-edge graph generation in this change.
- Do not add a multi-seed experiment matrix in this change. Variants reuse the main training config seed.
- Do not introduce a universal content-hash fingerprint framework. Existing manifest, run summary, effective config, and file checks remain the MVP status evidence.
- Do not migrate old run directories into the new ablation artifact layout.

## Decisions

### Decision 1: Keep the CLI entry point and introduce `scripts/workflow/`

The user-facing entry point remains `scripts/experiment.py`. New orchestration modules live under:

```text
scripts/
  experiment.py
  workflow/
    __init__.py
    types.py
    registry.py
    workflows.py
    artifacts.py
    manifest.py
    planner.py
    status.py
```

The package responsibilities are:

| Module | Responsibility |
| --- | --- |
| `types.py` | Closed enums and typed planning records. |
| `registry.py` | Static mapping from retrieval methods to workflow adapters and optional ablation suites. |
| `workflows.py` | Small lifecycle adapters for stateless retrieval, graph rerank, and R-GCN trainable retrieval. |
| `artifacts.py` | Deterministic artifact namespaces and reusable artifact aliases. |
| `manifest.py` | Manifest initialization, loading, config freezing, and legacy manifest compatibility. |
| `planner.py` | Matrix expansion, invalidation calculation, dependency validation, plan generation, and execution ordering. |
| `status.py` | Artifact evidence inspection and formatted status rows. |

`graph_memory/experiment.py` becomes a temporary compatibility facade for existing imports while orchestration logic moves into `scripts/workflow/`. The facade must not retain method-specific planning branches.

Alternative considered: keep adding helper functions to `graph_memory/experiment.py`. This would reduce file movement but would leave orchestration ownership unclear and continue growing a central script-like module.

Alternative considered: place workflow abstractions under `graph_memory/`. These abstractions coordinate scripts, manifests, and run directories rather than implementing retrieval algorithms, so `scripts/workflow/` is the clearer ownership boundary.

### Decision 2: Use explicit types for closed control vocabularies

Closed orchestration values use enums, including at least:

```text
StageId
WorkflowId
ArtifactRole
ChangeDimension
ArtifactState
RgcnAblationVariant
```

Public JSON and CLI surfaces still serialize their stable string values. Parsing converts strings into enums at the boundary and fails fast with the allowed values when input is unknown.

Method names remain registry-owned public identifiers because the supported method set grows by registration. Method discovery continues to list valid method names. Variant selection is validated against the selected method's registered suite, and the R-GCN suite exposes `RgcnAblationVariant`.

Alternative considered: use plain strings everywhere. This minimizes conversion code but hides valid choices from readers, weakens type checking, and makes planner branches easier to misspell.

Alternative considered: use one global enum for every future retriever and suite identifier. That would couple unrelated retrievers and make local workflow additions unnecessarily invasive.

### Decision 3: Separate runtime method construction from experiment workflow registration

`graph_memory.retrieval_registry` continues to own runtime retrieval construction metadata. `scripts/workflow/registry.py` owns experiment lifecycle registration:

```text
retrieval method
  -> workflow adapter
  -> optional ablation suite
```

The planner queries the workflow registry and does not infer lifecycle from `requires_checkpoint`, `requires_graphs`, or method-name membership checks.

Adding a method that uses an existing lifecycle requires a static registry entry and its runtime builder registration. Adding a genuinely new lifecycle requires one new workflow adapter plus its registry entry. No planner branch is added for either case.

Alternative considered: put lifecycle hooks directly on runtime retriever classes. Runtime retrievers should remain inference-focused: `task input -> ranked nodes + retrieved edges`. They must not know run directories, CLI scripts, training pairs, or experiment resume state.

### Decision 4: Model ablation as run-unit expansion, not as a stage

The planner expands selected methods into run units:

```text
bm25 / default
dense / default
dense_rgcn_graph_retriever / full_rgcn
dense_rgcn_graph_retriever / wo_bridge
dense_rgcn_graph_retriever / wo_entity_overlap
...
```

Each run unit follows its workflow's real lifecycle. The R-GCN workflow remains:

```text
prepare -> graphs -> pairs -> train -> retrieve -> evaluate -> aggregate
```

There is no opaque `ablate` stage in `StageId`. The CLI may filter ablation units, but plan output always exposes their actual `pairs`, `train`, `retrieve`, and `evaluate` commands.

Alternative considered: add one `ablate` stage that internally loops over variants. That would hide resumable work, obscure failures, and make downstream stage ranges ambiguous.

### Decision 5: Derive variant reuse from changed dimensions and workflow dependencies

Each `VariantSpec` declares:

- its stable variant identifier;
- its changed dimensions;
- its minimal effective-training-config overrides;
- whether it is the baseline alias.

Each workflow step declares:

- its stage identifier;
- semantic input artifact roles;
- output artifact roles;
- dimensions that invalidate the step;
- its low-level command adapter.

The planner calculates reuse as follows:

1. Find the first ordered workflow step whose invalidating dimensions intersect the variant's changed dimensions.
2. Alias all upstream artifact references to the main run unit.
3. Allocate variant-local paths for that step and all downstream method-specific outputs.
4. Intersect the resulting lifecycle with the user's requested stage range.
5. Fail fast if a required upstream artifact is neither scheduled nor already valid.

The first R-GCN suite uses:

| Variant | Changed dimension | Earliest invalidated step |
| --- | --- | --- |
| `full_rgcn` | none | aliases the main R-GCN run |
| `wo_bridge` | `MODEL_GRAPH_VIEW` | `train` |
| `wo_entity_overlap` | `MODEL_GRAPH_VIEW` | `train` |
| `wo_sequential` | `MODEL_GRAPH_VIEW` | `train` |
| `wo_query_overlap` | `MODEL_GRAPH_VIEW` | `train` |
| `wo_graph` | `MODEL_STRUCTURE` | `train` |
| `wo_edge_type` | `MODEL_STRUCTURE` | `train` |
| `wo_edge_weight` | `MODEL_STRUCTURE` | `train` |
| `wo_seed_score` | `MODEL_STRUCTURE` | `train` |
| `wo_hard_negatives` | `PAIR_SAMPLING` | `pairs` |

The suite includes the broader edge variants because the original experiment plan requires bridge and entity-overlap ablations and identifies sequential and query-overlap ablations as direct follow-ups. `random_edges` remains deferred because it changes graph construction and requires a separate graph transform or builder decision.

Alternative considered: let every variant directly name `start_from_stage`. That is simple but repeats workflow knowledge inside suite definitions and becomes fragile when a workflow changes.

Alternative considered: hard-code per-variant branches in the planner. That would solve the current R-GCN case but fail the extensibility goal.

### Decision 6: Maintain one main training config and layer minimal overrides

Users continue to maintain one R-GCN training config, normally:

```text
configs/training/dense_rgcn_graph_retriever/base.json
```

The experiment config adds:

```text
enable_ablation: false
ablation_variants:
  <method>: optional ordered subset of registered suite variants
```

When `enable_ablation` is true and no subset is specified, the planner expands all variants registered by the selected method's suite. CLI filters may narrow the selected variants for debugging or partial server runs.

For each non-baseline variant:

```text
effective variant config
  = resolved main training config
  + registered minimal variant override
```

The generated effective config is written under the variant directory for auditability. It is an output, not another hand-maintained input config.

The existing duplicated `configs/training/dense_rgcn_graph_retriever/ablations.json` template is retired or rewritten as documentation-facing selection metadata. It is no longer a second full training-config source.

Alternative considered: ask users to maintain one complete training config per variant. That would duplicate hyperparameters and make accidental drift likely.

### Decision 7: Use deterministic variant namespaces and explicit artifact aliases

Main outputs keep their current locations. Variant outputs use:

```text
runs/<experiment>/
  learned/<method>/...
  ablations/<method>/<variant>/
    effective_training_config.json
    train.pairs.json                     # only when invalidated
    train.pairs.summary.json             # only when invalidated
    train.pairs.run_summary.json         # only when invalidated
    train_metrics.jsonl
    train_run_summary.json
    checkpoints/
      best.pt
    predictions/
      test.ranked.json
    metrics/
      test.metrics.csv
    debug/
      failure_cases.jsonl
  config/
    ablation_metrics_index.json
  tables/
    ablation_results.csv
```

The manifest records concrete artifact references and alias sources. For example, `wo_graph` aliases the main `train.pairs.json`, while `wo_hard_negatives` owns a variant-local `train.pairs.json`.

`full_rgcn` is an identity alias to the main R-GCN unit. It does not schedule duplicate training, retrieval, or evaluation commands.

Alternative considered: create a separate top-level run name for each ablation. That works manually but makes one-command orchestration, aggregate tables, and resume status unnecessarily fragmented.

### Decision 8: Keep strict explicit resume semantics

Workflow adapters always describe a complete local lifecycle. They never inspect external planner state or decide whether a stage should be skipped.

The planner combines:

```text
full workflow
+ requested stage range
+ resolved artifact aliases
+ current artifact evidence
= concrete plan
```

Examples:

- A default run with ablation enabled schedules the main workflow and all missing variant-local work.
- `run <name> --from retrieve` schedules variant `retrieve -> evaluate -> aggregate` only when each required checkpoint is already valid.
- If a required variant checkpoint is missing, `--from retrieve` fails fast and reports the missing path. It does not silently insert `train`.
- `--ablations-only` excludes ordinary method commands except shared prerequisites needed by selected variants.
- Repeated `--variant <name>` filters support smoke tests and partial server runs.

The CLI also adds read-only ablation discovery:

```text
scripts/experiment.py ablations list
scripts/experiment.py ablations list --method dense_rgcn_graph_retriever
```

Alternative considered: allow downstream commands to silently backfill missing upstream work. That makes user intent and server cost difficult to audit.

### Decision 9: Aggregate ablation results through an explicit run-local index

`scripts/aggregate_tables.py` remains the low-level table adapter. When ablation is enabled, the planner supplies:

```text
--ablation_index runs/<experiment>/config/ablation_metrics_index.json
--output_ablation runs/<experiment>/tables/ablation_results.csv
```

The index lists the method, variant, and concrete metric CSV path for each row, including the aliased `full_rgcn` metrics path. The aggregate script does not need to understand workflow rules or infer variants from path names.

The ablation table includes:

```text
Method
Variant
Recall@5
Full Support@5
Connected Evidence Recall@10
Path Recall@10
Retrieval Latency / Query
```

Existing connectivity metrics keep their current reference-graph meaning so results remain comparable across methods and variants. R-GCN prediction artifacts will return only model-visible retrieved edges for edge-view variants, ensuring debug artifacts and `Avg Retrieved Edges` remain faithful to the model-visible graph. A separate model-visible connectivity table is deferred unless later analysis demonstrates that it is needed.

Alternative considered: recursively scan variant directories and infer the variant name from path structure. An explicit index is more reviewable and avoids coupling aggregation to directory parsing.

### Decision 10: Preserve legacy runs and use a manifest schema revision

Newly initialized manifests use a revised schema that records run units, variant specs, artifact aliases, and ablation-table paths. Existing schema-version-1 manifests remain readable as non-ablation runs.

Enabling ablation for an already initialized schema-version-1 run requires a new run name or explicit reinitialization. The runner does not rewrite existing artifact trees in place.

Alternative considered: migrate old manifests and artifact trees automatically. The migration adds risk without helping current experiment execution.

## Risks / Trade-offs

- [Risk] `scripts/workflow/` becomes an internal framework with too many generic hooks. -> Mitigation: support only ordered project workflows, typed artifact roles, and static in-code registries; do not build a DAG DSL or plugin loader.
- [Risk] Runtime registry and workflow registry drift apart. -> Mitigation: add registry validation tests that every public retrieval method has exactly one workflow registration and every suite references a registered method.
- [Risk] Generated variant configs accidentally diverge from the main training hyperparameters. -> Mitigation: generate configs by merging only registered minimal overrides and assert unchanged sections in tests.
- [Risk] Edge-view variants report original induced edges even though the model did not see them. -> Mitigation: apply the model-visible edge policy when writing R-GCN retrieved subgraphs and test the returned edges.
- [Risk] A user expects `--from retrieve` to repair missing checkpoints automatically. -> Mitigation: retain explicit fail-fast behavior and print the missing variant artifact paths.
- [Risk] The first suite becomes larger than needed for a smoke test. -> Mitigation: support repeated `--variant` filters while keeping the full registered suite discoverable.
- [Risk] Compatibility facade persists indefinitely. -> Mitigation: keep it thin, test that planning lives in `scripts/workflow/`, and document it as transitional.

## Migration Plan

1. Add failing tests for typed workflow registrations, invalidation boundaries, aliases, config expansion, deterministic variant paths, strict downstream resume, discovery, and ablation aggregation.
2. Introduce `scripts/workflow/` typed records, registries, and lifecycle adapters while preserving existing non-ablation runner outputs.
3. Move manifest, artifact-layout, planner, and status responsibilities out of `graph_memory/experiment.py`; retain a thin compatibility facade.
4. Add `enable_ablation` config handling and R-GCN suite definitions with minimal variant overrides.
5. Add missing R-GCN edge-view variants and `wo_hard_negatives`; keep their low-level training and inference paths shared with the full model.
6. Add variant artifact allocation, alias recording, plan expansion, strict resume validation, and readable variant-qualified plan output.
7. Extend aggregation with the explicit ablation metric index and write `tables/ablation_results.csv`.
8. Update config and operations documentation, including removal or retirement of the duplicated full ablation training-config template.
9. Run targeted tests, the full test suite, lint, diff checks, and OpenSpec validation.

Rollback is straightforward: disable `enable_ablation` and retain the legacy facade until non-ablation runner compatibility is verified. Low-level scripts remain directly runnable throughout the migration.

## Open Questions

No blocking design questions remain for implementation planning. Multi-seed expansion, random-edge graph construction, and a separate model-visible connectivity table are intentionally deferred.

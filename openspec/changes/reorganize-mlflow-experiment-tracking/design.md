## Context

The current workflow correctly preserves scientific artifacts and typed stage summaries, but its MLflow mapping exposes execution internals. Hydra uses the complete override dirname as the job leaf, the parent recursively publishes the entire resolved configuration, and each executed stage attempt creates a child. Evaluation and aggregate CSV readers then log method-prefixed numeric cells as independent metrics. The result is reproducible but difficult to inspect or compare.

The local workflow remains the authority: `WorkflowPlanner` builds a stage-major plan, subprocess stages write local artifacts and summaries, status/resume validates those artifacts, and MLflow is a strict mirror. This change must improve the mirror without reordering stages, changing model code, weakening tracking-failure propagation, or making MLflow part of cache correctness.

The completed but unarchived workflow changes currently specify one child per stage attempt. This change intentionally replaces that presentation contract. It does not add a compatibility layer because the user has explicitly rejected legacy branches, migration, and version tags.

## Goals / Non-Goals

**Goals:**

- Give multirun jobs concise, deterministic filesystem and MLflow names such as `0_num_layers=2`.
- Make one Hydra job parent a small experiment summary with an all-baseline Overview table and organized total artifacts.
- Make one user-visible baseline identity one MLflow child containing its method parameters, final metrics, training series when applicable, and baseline artifacts.
- Make final metric keys identical across baseline children so standard MLflow Compare Runs is the comparison surface.
- Remove stage-child noise, duplicate aggregate/evaluate metrics, and operational observations from the model-metric namespace.
- Preserve direct scripts, local stage summaries, cache/resume, failure semantics, shared SQLite paths, and the large-artifact prohibition.

**Non-Goals:**

- Logging comparison metrics, training series, or generated comparison plots on parents.
- Generating repository-owned comparison images or installing saved MLflow chart layouts.
- Customizing or forking the MLflow frontend to suppress its native scalar charts.
- Reordering the stage-major workflow into a method-major workflow.
- Changing metric definitions, training logs, evaluation CSV schemas, aggregation outputs, cache rules, or delivery artifacts.
- Migrating, rewriting, tagging, dual-writing, or automatically resuming runs created under the former stage-child organization.
- Adding a tracking schema/version tag or any compatibility reader.

## Decisions

### 1. `layout.py` owns one concise multirun leaf used by both Hydra and repository artifacts

The concise suffix is derived from Hydra's job-discriminating overrides after configured root identity keys such as `name`, `dataset`, `profile`, and `methods` are excluded. Nested keys collapse to their final component, so `method_configs.dense_rgcn_graph_retriever.train.model.num_layers=2` becomes `num_layers=2`. Multiple remaining assignments retain deterministic Hydra order. Duplicate leaf keys are rejected before writing outputs.

Hydra's `sweep.subdir` and `RunLayout.run_dir` must call the same pure formatter; the current equality check between Hydra's output directory and `RunLayout` remains. Implementation begins with a real Hydra spike proving that a registered OmegaConf resolver can format `hydra.job.override_dirname` before any production change. A separate Hydra output tree and repository run tree is rejected because it would recreate split ownership.

The full override strings remain in `config/overrides.yaml`, the resolved config remains in `config/resolved.yaml`, and `multirun.yaml` remains the sweep catalog. The directory leaf is an identifier, not a serialization format.

Alternative considered: use only `0`, `1`, `2`. Rejected because the user wants the changed value visible. Alternative considered: retain the full override dirname. Rejected because it obscures the sweep dimension and creates fragile paths.

### 2. Parent tracking is curated and contains no native result metrics

`TrackingAdapter` replaces recursive `config.normalized()` publication with an explicit parent parameter map owned in `tracking.py`. It contains only job identity and experiment selection fields. The complete resolved YAML is uploaded under `config/` rather than repeated across hundreds of parameters.

After aggregate succeeds, tracking renders the existing aggregate tables into the parent `mlflow.note.content` description as one readable all-baseline result table. The same source CSVs are uploaded under `results/`. Shared prepare/graph summaries and the aggregate summary are uploaded under `workflow/`. A stage-bounded run without aggregate output receives a concise textual summary and available artifacts but no fabricated result table.

Parent result metrics are deliberately absent. This makes baseline children, not parents, the unit selected in MLflow Compare Runs. Parent-generated comparison images and saved chart configuration are also absent.

Alternative considered: duplicate final metrics on the parent for multirun comparison. Rejected by the revised responsibility boundary. Alternative considered: generate comparison PNG/SVG artifacts. Rejected because comparison belongs to MLflow Compare Runs.

### 3. Execution owns a direct baseline-child lifecycle map; planning remains stage-major

`WorkflowPlanner` remains unchanged. `execution.py` owns a small mapping from baseline identity `(method, variant)` to one MLflow child ID and uses direct `TrackingAdapter` operations to create, update, and terminate those children. No generic event bus, hook system, provider interface, or second workflow abstraction is introduced.

Stage observations are routed as follows:

| Local stage | MLflow owner |
|---|---|
| `prepare:*`, `graphs:*` | parent workflow artifacts and status |
| `pairs:<method>`, `tune:<method>`, `train:<method>` | baseline child |
| `retrieve:<method>`, `evaluate:<method>` | baseline child |
| `aggregate:aggregate` | parent Overview and result artifacts |

A baseline child is created when its first owned observation is available. On a successful full run, finalization creates/populates any selected baseline child whose stages were entirely cached, so child count reflects the selected baselines rather than cache misses. Exact resume finds the unique child by parent ID plus method/variant tags and reuses it; duplicate matches are a strict error. Local cache validation never queries MLflow.

Because the plan is stage-major, several baseline children can remain RUNNING while later stages for other baselines execute. The last owned stage or finalization terminates each child. A method-owned failure fails its child and parent; a shared-stage failure fails the parent without inventing untouched baseline children.

Alternative considered: reorder execution to complete one baseline at a time. Rejected because it changes dependency order and cache/runtime behavior. Alternative considered: create baseline children only after aggregation. Rejected because failures would lose useful partial training observations.

### 4. User selection, not hidden planner expansion, defines baseline children

The public `config.methods` plus explicit ablation variants define user-visible baseline identities. Expanded train dependencies do not automatically become children. When a hidden Dense-FT training dependency exists only to seed `dense_ft_rgcn_graph_retriever`, its summaries and artifact metadata are attached below that selected child's `dependencies/dense_ft/` namespace. When Dense-FT is also explicitly selected, it owns its own child and the dependent child contains only a reference to the selected Dense-FT artifacts.

This preserves the invariant that seven selected baselines produce seven children without hiding the prerequisite execution.

### 5. Final metric keys are method-independent and logged once

`tracking.py` owns one explicit mapping from the repository's human-readable evaluation/efficiency column names to concise MLflow keys:

```text
Recall@2                              -> final.recall_at_2
Recall@5                              -> final.recall_at_5
Recall@10                             -> final.recall_at_10
Evidence F1@5                         -> final.evidence_f1_at_5
Evidence F1@10                        -> final.evidence_f1_at_10
Full Support@5                        -> final.full_support_at_5
Full Support@10                       -> final.full_support_at_10
MRR                                   -> final.mrr
Connected Evidence Recall@5           -> final.connected_evidence_recall_at_5
Connected Evidence Recall@10          -> final.connected_evidence_recall_at_10
Query-Evidence Connectivity@10        -> final.query_evidence_connectivity_at_10
Path Recall@10                        -> final.path_recall_at_10
Edge Recall@10                        -> final.edge_recall_at_10
Retrieval Latency / Query             -> final.retrieval_latency_per_query
Index Build Time                      -> final.index_build_time
Graph Construction Time               -> final.graph_construction_time
Memory Size                           -> final.memory_size
Avg Retrieved Nodes                   -> final.avg_retrieved_nodes
Avg Retrieved Edges                   -> final.avg_retrieved_edges
```

The method and variant are child tags and parameters, never part of these keys. Applicable numeric values are logged once from the baseline's authoritative evaluation/aggregate row. `N/A` values remain in tables and are omitted from MLflow numeric metrics. Aggregate tracking never logs the same row again.

This mapping intentionally means MLflow may render each final scalar as a single-point chart on an individual child. That native behavior is accepted because Compare Runs requires queryable metrics. The repository removes avoidable duplication and pseudo-metrics but does not modify MLflow UI behavior.

### 6. Only genuine epoch observations are training metrics

Trainable children reuse the existing `train_metrics.jsonl` parser and the epoch as step. Numeric record fields keep stable `train.*` keys. Nested maps, strings, booleans, null, NaN, and infinity are not metrics. Counts, durations, artifact sizes, stage status, and exception text move to tags, parameters, Overview text, or stage-summary artifacts.

Final evaluation scalars use `final.*`; training/development histories use `train.*`. This makes the distinction searchable and prevents aggregate table ingestion from creating an unbounded metric namespace.

### 7. Artifacts are organized by research responsibility without duplicating large data

Existing local files are uploaded or referenced under stable MLflow artifact paths:

```text
parent
  config/resolved.yaml
  config/overrides.yaml
  results/main_results.csv
  results/path_results.csv
  results/efficiency_results.csv
  results/ablation_results.csv       # when applicable
  workflow/<shared-stage summaries>

baseline child
  config/method.yaml
  tuning/<selected config and candidate table>
  training/train_metrics.jsonl
  training/<train summary>
  evaluation/<metric table and summary>
  workflow/<method-stage summaries>
  dependencies/<hidden dependency summaries>
```

Small training metric and evaluation files become eligible curated artifacts subject to the existing size ceiling. Datasets, graphs, pairs, predictions, checkpoints, and model directories remain prohibited uploads and retain path/kind/size/role metadata only.

### 8. Cutover is direct and old data is historical

No database migration or tracking tag identifies old versus new records. Existing SQLite rows and old run directories remain untouched. After cutover, new experiments must use a fresh name or explicitly reset an old local named run before execution. Reusing an old stage-child parent is not supported and receives no detection, conversion, or dual-write logic.

This is intentionally operational guidance rather than compatibility code.

## Risks / Trade-offs

- [MLflow still renders final scalars as single-point charts] → Accept native behavior; log each final metric once, eliminate aggregate duplicates and pseudo-metrics, and use Compare Runs for analysis.
- [Stage-major execution keeps several baseline children open concurrently] → Keep lifecycle ownership in `execution.py`, terminate every open child on parent failure, and cover mixed success/failure order with integration tests.
- [A cached baseline has no newly executed stage event] → Populate/finalize its child from authoritative local summaries and metric artifacts after resume planning.
- [A concise leaf can collide when nested keys share a name] → Reject duplicate collapsed leaf keys before output creation.
- [Parent Overview table rendering can drift across MLflow versions] → Add a locked MLflow 3.14 runtime spike and preserve CSV artifacts as the durable table even if description formatting is limited.
- [Old named runs can form mixed projections if reused manually] → Document fresh names or explicit reset; do not add automatic detection or compatibility behavior.
- [Final metric mapping can drift from evaluation columns] → Test the explicit mapping against the canonical metric/result contracts and fail on an unknown numeric final column rather than silently dropping it.
- [Tracking changes could accidentally affect cache truth] → Keep planner, artifact validation, summaries, and resume decisions unchanged and assert that no tracking query participates in cache decisions.

## Migration Plan

1. Add failing layout and real Hydra multirun tests for concise job leaves and ambiguity rejection.
2. Replace the parent parameter/artifact projection and prove parent metrics remain empty.
3. Introduce baseline-child routing and lifecycle while leaving planner and subprocess stages unchanged.
4. Replace dynamic CSV numeric mirroring with the explicit final metric mapping and genuine epoch series.
5. Organize parent and child artifacts and render the parent Overview table.
6. Update status/inspect/reset job parsing for concise leaves without adding old-leaf fallback logic.
7. Run fresh-name single-run, seven-baseline, trainable, cache-hit, resume, failure, ablation, and multirun acceptance workflows against a disposable tracking database before using the shared store.
8. Cut over directly. Existing shared-store rows remain untouched; users use new names or explicitly reset old local runs.

Rollback is a normal source rollback before new production runs are adopted. No database rollback or reverse migration is provided because this change does not rewrite existing MLflow records.

## Open Questions

None. The remaining MLflow 3.14 Description rendering and Hydra resolver checks are implementation spikes with specified fallbacks and acceptance criteria, not product decisions.

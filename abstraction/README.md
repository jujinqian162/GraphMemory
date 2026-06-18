# Cross-Dataset Abstraction Skeleton

This directory is a pseudo-code planning artifact for
`docs/30-design/cross-dataset-refactor-design.md`.

It is intentionally separate from the runnable `graph_memory` package. The
files describe domain boundaries, naming, ports, adapters, and composition
flow. Method bodies are intentionally left as `pass` so the first review can
focus on architecture and logic instead of implementation details.

Directory intent:

- `domain/datasets`: raw dataset ownership, assets, official splits, recipes.
- `domain/task_views`: stable benchmark task views and eval label views.
- `domain/projections`: anti-corruption adapters between views, requests, and
  evaluation units.
- `domain/retrieval`: request ports, prediction views, method capabilities.
- `domain/graphs`: graph build artifacts, rule sets, graph construction port.
- `domain/evaluation`: metric suite ports and metric result records.
- `domain/training`: trainable-method ports and training artifact contracts.
- `domain/scripts`: independently reproducible step scripts and CLI branches.
- `domain/workflow`: stage graph planning and orchestration of script commands.

Boundary rule:

- Script modules own the high-level logic inside a step, such as dataset
  preparation, request projection, graph building, training, retrieval, and
  evaluation.
- Script CLI entrypoints model top-level files such as `run_retrieve_stage.py`.
  They parse/select branches in `run(args)` and build dependencies through
  `ScriptLocalCompositionRoot`, not through external constructor injection.
- Script CLI entrypoints after `prepare_dataset` consume upstream intermediate
  artifacts from `ScriptCommand.read_artifacts`. They do not rerun previous
  scripts. For example, `RunBuildGraphScript` loads prepared task-view
  artifacts, and `RunRetrieveStageScript` loads a projected request artifact.
- Retrieval script branches are meaningful method-family lanes. Text ranking
  calls the text-ranking step, trainable text ranking also loads a training
  artifact, graph ranking loads graph artifacts, temporal-memory ranking loads
  sidecar artifacts, and context gathering optionally loads graph context.
- Retrieval steps do not hide method execution behind a generic boundary. They
  assemble a typed request through `RetrievalRequestAssembler`, resolve the
  runtime method through `MethodRuntimeResolver`, then call the method protocol
  such as `rank_task`, `gather_task_context`, or `answer_task`.
- Script step return values use direct `Result` names. Small stage outputs are
  tuple aliases instead of named bundles unless the fields need a real domain
  owner.
- Step classes are script-internal actions. They do not compose all scripts
  together; workflow owns cross-script composition.
- Workflow modules own the experiment-level command plan: which script command
  runs, in what order, with which arguments and artifact dependencies.

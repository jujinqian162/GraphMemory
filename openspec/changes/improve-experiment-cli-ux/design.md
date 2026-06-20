## Context

`scripts/experiment.py` is the public entrypoint for named runs, while `graph_memory.experiment` owns manifest creation, stage planning, command generation, status inspection, and config loading. The current runner already centralizes artifact paths under `runs/<experiment_name>/` and keeps low-level scripts explicit, but the user-facing interface still leaks implementation detail: users must know which stages belong to each method and must discover configs or methods by reading files/code.

The existing contracts to preserve are: named run isolation, manifest reuse, `CLI overrides > experiment config/profile > code defaults`, low-level script compatibility, fail-fast validation, and the lightweight retrieval method registry.

## Goals / Non-Goals

**Goals:**

- Make the common path `run <name> --method <method>` plan and execute the complete required workflow for that method.
- Represent stage selection as a contiguous range over the selected method workflow when `--from`/`--to` is used.
- Retire explicit stage-list selection from the public runner contract; use `--from`/`--to` for partial runs and the selected method workflow for defaults.
- Add resource discovery subcommands so users can list stages, methods, configs, profiles, and recipes from the CLI.
- Resolve top-level experiment configs and method training configs by contract names, while still accepting explicit paths for compatibility.
- Improve `plan` rendering so generated commands are scan-friendly and optional terminal color is isolated to presentation.

**Non-Goals:**

- Do not rewrite low-level data preparation, graph construction, retrieval, evaluation, training, or aggregation scripts.
- Do not introduce dynamic plugin loading for methods; the static registry remains the source of truth.
- Do not remove `--methods` in this change.
- Do not change metric definitions, artifact filenames, or run directory layout.

## Decisions

1. Method workflows are derived from registry metadata instead of hard-coded CLI branches.

   Each selected method maps to required stages by reading `RetrievalMethodSpec` flags. All methods require `prepare`, `graphs`, `retrieve`, `evaluate`, and `aggregate` because evaluation currently consumes graph artifacts. Graph-rerank methods add `tune`; checkpoint-backed methods add `pairs` and `train`. Multi-method runs use the ordered union of required stages, preserving `STAGE_ORDER`.

   Alternative considered: add a separate workflow registry for every method. That would be explicit but duplicates the method registry and increases the chance of stale method/stage contracts.

2. `--from`/`--to` select ranges over the selected method workflow.

   For `--method bm25 --from prepare --to retrieve`, the selected stages are `prepare`, `graphs`, `retrieve`. For `--method dense_rgcn_graph_retriever --from prepare --to retrieve`, they are `prepare`, `graphs`, `pairs`, `train`, `retrieve`. If a bound is not part of the selected method workflow, the runner fails fast and prints the valid stage names.

   Alternative considered: slice raw `STAGE_ORDER`. That would reintroduce irrelevant stages into simple method runs and weaken the method-first contract.

3. Trainable methods compose required artifacts rather than masquerading as graph-rerank methods.

   `dense_rgcn_graph_retriever` depends on the same prepared tasks and graph artifacts, plus train-pair and checkpoint artifacts under `runs/<experiment>/learned/<method>/`. It does not require `dense_graph_rerank` tuned output unless a future registry flag explicitly says so. This keeps the branch/merge style composition at the artifact-contract level instead of by implicit method-name coupling.

   Alternative considered: automatically run `dense_graph_rerank` before the R-GCN method. That would produce unrelated predictions and tuned configs today, making the workflow slower and less explainable.

4. Discovery subcommands read from the same sources execution uses.

   `stages list` reads stage metadata, `methods list` reads `graph_memory.retrieval_registry`, `configs list` scans the known contract directories, `profiles list` loads an experiment config, and `recipes list` summarizes experiment config files. These commands are read-only and do not create manifests.

   Alternative considered: maintain separate documentation-only lists. That improves prose but does not solve stale CLI discoverability.

5. Config names are resolved at boundaries.

   `--config hotpotqa_evidence_retrieval` resolves to `configs/experiments/hotpotqa_evidence_retrieval.json`; explicit paths remain supported. Training config references may be existing paths, `base`, or `<method>/base`, resolving under `configs/training/<method>/`. Search-space paths remain persisted as low-level command paths, but discovery exposes their contract names.

   Alternative considered: rewrite all config JSON to store only names. That would be a broader migration and is unnecessary while the loader can support both names and paths.

6. Plan formatting is a presentation concern.

   `format_commands` will render each command as a block with index, stage, qualifier, script file, and one option per line. ANSI color is only applied to option names when requested or when stdout is a terminal, leaving stored command argv unchanged.

## Risks / Trade-offs

- Existing automation may parse one-command-per-line `plan` output -> keep a structured `StageCommand.argv` API and only change the human-facing CLI formatting; tests pin the new output.
- Existing automation that still passes an explicit stage list will fail argument parsing -> update current docs and tests to point to default workflow execution or `--from`/`--to` ranges.
- Config-name resolution could hide typos if it falls back too broadly -> use deterministic directories and fail with the attempted contract path.
- Multi-method ranges may include a stage needed by only one method -> commands are generated only for methods that require the stage, preserving ordered union behavior.

## Context

Phase 1 has deliberately strict low-level scripts: every script receives explicit input and output paths, validates artifact contracts, and writes run summaries. That design is correct for leakage safety and code review, but it makes normal experiment operation awkward. A complete run requires many long commands, repeated path construction, and manual discipline to avoid mixing artifacts from different experiments.

The original project plan also extends beyond Phase 1. Later work can add Dense-FT, Memory Stream, GraphRAG, ablations, 2WikiMultiHopQA, MuSiQue, and tool trajectories. The high-level runner should therefore be named around experiments, not around Phase 1, while still avoiding a generic pipeline framework before the project needs one.

## Goals / Non-Goals

**Goals:**

- Provide a single user-facing experiment entry point for running, planning, resuming, and inspecting experiment workflows.
- Keep existing low-level scripts as explicit artifact-contract adapters.
- Isolate all run-specific artifacts under `runs/<experiment_name>/`.
- Generate and maintain a manifest that records the effective config, artifact paths, stage selections, method selections, and provenance checks for a run.
- Clarify config ownership by separating stable experiment defaults, tuning search spaces, published selected configs, and run-local outputs.
- Implement the current HotpotQA evidence-retrieval workflow without baking `phase1` into the runner concept.
- Leave room for later recipes and stages without dynamic plugin discovery.

**Non-Goals:**

- Do not change low-level CLI argument names or artifact schemas.
- Do not implement Phase 2/3 methods, new datasets, training stages, or ablation stages in this change.
- Do not introduce a general workflow engine, external scheduler, plugin registry, or YAML-defined command language.
- Do not make global config mutable during ordinary experiment runs.
- Do not remove the low-level command documentation used for debugging and contract review.

## Decisions

### Decision 1: Name the user entry point `experiment`

Use a script named `scripts/experiment.py` rather than `phase1_workspace.py`.

The domain concept is an experiment run, not a phase-specific workspace. A phase is a recipe choice; a workspace/run is the isolated place where artifacts are produced. This keeps the user-facing model stable when later phases add new methods and datasets.

Alternative considered: `phase1_workspace.py`. That describes the current implementation moment, but it would make the first abstraction obsolete as soon as Phase 2 starts.

Alternative considered: `run_pipeline.py`. That is too generic and suggests a pipeline engine rather than a small project-specific experiment runner.

### Decision 2: Keep low-level scripts as the contract boundary

The runner should call existing scripts such as `prepare_hotpotqa.py`, `build_graphs.py`, `run_retrieval.py`, `tune_graph_rerank.py`, `evaluate_retrieval.py`, and `aggregate_tables.py` with explicit generated paths.

This preserves the current contract-first design: scripts still declare their input and output artifacts, validators still run at script boundaries, and run summaries still explain what happened. The runner removes path burden from the user; it does not hide artifact contracts from the system.

Alternative considered: refactor all low-level scripts into a single Python service and bypass their CLI surfaces. That would reduce subprocess overhead but weaken the reviewable command boundary and make the change larger than necessary.

### Decision 3: Use a run manifest as the orchestration source of truth

Each run directory should contain `manifest.json`. The manifest should include at least:

```text
schema_version
experiment_name
recipe
profile
created_at / updated_at
effective_config
selected_methods
selected_stages
artifact paths by split, method, and stage
stage status metadata
fingerprint or provenance fields for key inputs/configs
```

The manifest owns generated paths. Users should not hand-name prediction files, tuned config files, metric files, or aggregate tables through the high-level runner.

Alternative considered: derive all paths dynamically from config every time. That is simpler initially, but it makes resume/status behavior weaker because the system lacks a durable record of what the run originally meant.

### Decision 4: Use fixed run-local artifact layout

Use this first layout:

```text
runs/
  <experiment_name>/
    manifest.json
    config/
      effective_config.json
    inputs/
    graphs/
    tuned/
    predictions/
    metrics/
    tables/
    debug/
    summaries/
```

The exact filenames inside these directories should be deterministic, based on split and method names. Run-local tuned configs belong in `runs/<experiment>/tuned/`, not in global `configs/`.

Alternative considered: keep writing to `data/hotpotqa/processed` and `results`. That preserves old examples but keeps exploratory and official artifacts mixed together.

### Decision 5: Separate config roles by directory

Use clear config locations:

```text
configs/experiments/
configs/search_spaces/
configs/published/
```

`configs/experiments/` stores stable experiment defaults such as dataset paths, split policies, graph settings, dense encoder settings, default methods, and profiles.

`configs/search_spaces/` stores tuning grids such as graph-rerank parameter candidates.

`configs/published/` stores curated configs selected for paper or reproduced results. Ordinary tuning writes selected configs into the run directory.

Alternative considered: keep `phase1_default.json` and `phase1_graph_rerank_grid.json`. Their names do not communicate whether they are experiment defaults, search spaces, or selected results, and they make later phases harder to organize.

### Decision 6: Implement a small stage recipe, not a generic engine

The first runner should have a small in-code recipe for the current evidence-retrieval workflow:

```text
prepare -> graphs -> tune -> retrieve -> evaluate -> aggregate
```

The recipe should define stage dependencies, expected inputs, expected outputs, and the low-level command to run. Future work can add `pairs`, `train`, `ablate`, `generalize`, or `case-study` stages by extending the recipe, but this change should not create plugin discovery or a broad workflow DSL.

Alternative considered: model every current and future phase in a declarative DAG config. That is premature for the current project and would make simple runs harder to debug.

### Decision 7: Treat status checks as provenance checks, not only file-exists checks

Stage completion should consider expected artifact existence plus run summaries or manifest metadata. When possible, the runner should detect that an artifact was produced from a different input path, method, split, or effective config and report it as stale or mismatched.

Alternative considered: skip completed stages whenever output files exist. That is fast but fails the main safety goal: preventing accidental cross-run artifact reuse.

## Risks / Trade-offs

- [Risk] The runner could become a second hidden pipeline with behavior that diverges from low-level scripts. -> Mitigation: generate low-level commands from explicit stage recipes, support a dry-run plan command, and keep low-level command docs.
- [Risk] Config layering can become confusing. -> Mitigation: restrict precedence to `CLI overrides > experiment config/profile > code defaults`, write `effective_config.json`, and record the same config in the manifest.
- [Risk] Future phases may require stages that do not look like current retrieval stages. -> Mitigation: keep the runner recipe extension point small and in-code; add new recipes only when those phases are implemented.
- [Risk] Moving common outputs under `runs/` may confuse users familiar with `data/processed` and `results`. -> Mitigation: document `runs/` as the recommended path and keep low-level examples for canonical/debug runs.
- [Risk] Provenance fingerprints can overreach and block legitimate resumes. -> Mitigation: start with clear path/config/method checks and report actionable status; add stronger hashes only where needed.

## Migration Plan

1. Add runner tests for manifest creation, deterministic path generation, stage planning, method selection, resume behavior, and stale artifact detection.
2. Add the first experiment config under `configs/experiments/` and the graph-rerank search space under `configs/search_spaces/`.
3. Implement `scripts/experiment.py` and small orchestration helpers that generate low-level commands without changing existing scripts.
4. Route run-specific outputs into `runs/<experiment>/` and keep tuned graph configs run-local by default.
5. Update docs so `scripts/experiment.py` is the recommended path and low-level commands are documented as contract/debug commands.
6. Keep existing config files during the first migration only if tests or docs still reference them, then remove or replace ambiguous names once references are updated.

Rollback is straightforward because low-level scripts and artifact contracts remain unchanged. Users can continue running the existing command sequence directly if the high-level runner has a defect.

## Open Questions

- Should the command eventually be exposed as a console entry point such as `graphmem`, or is `python scripts/experiment.py` sufficient for now?
- Should official published selected configs be created immediately, or only after a full verified experiment run exists?

# Memory Stream Global Importance Prepare Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the run-local importance workflow stage with a zero-argument, one-time global preprocessing command whose artifact can be reused by every workflow workspace.

**Architecture:** Keep prompt, cache, runtime, and annotation logic in the Memory Stream package. Move annotation settings out of the workflow stage-config registry, let the standalone CLI own all IO paths and defaults, and add a subset-safe consumer validator for later retrieval integration. Restore workflow code to its pre-Memory-Stream state until method implementation resumes.

**Tech Stack:** Python dataclasses and argparse, typed JSON contracts, SHA-256 content-addressed cache, PyTorch/Transformers lazy loading, pytest, Ruff, BasedPyright, OpenSpec.

---

### Task 1: Freeze Standalone CLI Defaults

**Files:**
- Modify: `tests/test_memory_stream_annotation_stage.py`
- Modify: `tests/test_cli_contracts.py`
- Modify: `scripts/annotate_importance.py`
- Create: `graph_memory/retrieval/methods/memory_stream/settings.py`

- [ ] Add a test asserting `parse_args([])` resolves the canonical dev task,
  global output, derived summary, cache, model id/path, device, prompt version,
  and token defaults.
- [ ] Add a zero-argument CLI smoke using a temporary working directory,
  canonical default paths, and an injected fake runtime.
- [ ] Run the focused tests and confirm they fail because the current CLI
  requires `--config`.
- [ ] Implement `ImportanceAnnotationSettings` in the Memory Stream package and
  implement the standalone argparse adapter.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Support Global Artifact Subset Consumption

**Files:**
- Modify: `tests/test_memory_stream_importance.py`
- Modify: `graph_memory/validation/importance.py`
- Modify: `graph_memory/validation/__init__.py`

- [ ] Add tests proving a canonical global artifact can select workflow tasks in
  a different order while allowing extra artifact tasks.
- [ ] Add failure tests for missing and duplicate artifact task ids and changed
  content digests.
- [ ] Run the focused tests and confirm the subset API is absent.
- [ ] Implement an indexed subset-selection validator that returns records in
  requested task order while preserving strict per-task validation.
- [ ] Run the focused tests and confirm they pass.

### Task 3: Remove Workflow and Stage-Config Ownership

**Files:**
- Modify: `graph_memory/registry/ids.py`
- Modify: `graph_memory/registry/stage_configs.py`
- Delete: `graph_memory/stages/importance.py`
- Modify: `scripts/workflow/contracts.py`
- Modify: `scripts/workflow/manifest.py`
- Modify: `scripts/workflow/planner.py`
- Modify: `scripts/workflow/registry.py`
- Modify: `scripts/workflow/stage_configs.py`
- Modify: `scripts/workflow/types.py`
- Modify: `scripts/workflow/workflows.py`
- Modify: `configs/experiments/hotpotqa_evidence_retrieval.json`
- Delete: `tests/test_memory_stream_workflow.py`
- Modify: workflow and registry contract tests

- [ ] Add or restore tests asserting no workflow `importance` stage, run-local
  artifact mapping, annotation stage config, or planner command exists.
- [ ] Run workflow contract tests and confirm they fail against the current
  integration.
- [ ] Remove the annotation stage id/config registry and all workflow producer
  integration.
- [ ] Restore experiment configuration to contain no annotation IO or model
  lifecycle settings.
- [ ] Run workflow and registry contract tests and confirm they pass.

### Task 4: Remove Premature Method Registration

**Files:**
- Modify: `graph_memory/registry/methods.py`
- Modify: `graph_memory/registry/retrieval.py`
- Modify: `graph_memory/registry/stage_configs.py`
- Modify: relevant registry and orchestration tests

- [ ] Restore tests showing `memory_stream` is not yet a public retrieval method
  during this prepare-only milestone.
- [ ] Remove the premature retrieval settings, lifecycle, importance dependency,
  and retrieve IO field introduced with workflow integration.
- [ ] Retain annotation contracts, prompt, cache, runtime, validation, and atomic
  IO support.
- [ ] Run registry and annotation tests.

### Task 5: Correct OpenSpec and Maintainer Plan

**Files:**
- Modify: `openspec/changes/add-memory-stream-retrieval/proposal.md`
- Modify: `openspec/changes/add-memory-stream-retrieval/design.md`
- Modify: `openspec/changes/add-memory-stream-retrieval/specs/memory-stream-importance-annotation/spec.md`
- Modify: `openspec/changes/add-memory-stream-retrieval/specs/memory-stream-experiment-workflow/spec.md`
- Modify: `openspec/changes/add-memory-stream-retrieval/specs/memory-stream-retrieval/spec.md`
- Modify: `openspec/changes/add-memory-stream-retrieval/tasks.md`
- Modify: `docs/10-plans/memory-stream-implementation-plan.md`

- [ ] Replace first-class workflow-stage language with global one-time
  preprocessing ownership.
- [ ] Specify the zero-argument defaults and subset-safe external dependency.
- [ ] Mark only completed prepare work complete; leave retrieval and workflow
  consumer integration pending.
- [ ] Run `openspec validate add-memory-stream-retrieval --strict`.

### Task 6: Verify, Commit, and Push

**Files:**
- Review all modified files.

- [ ] Run focused annotation, CLI, registry, and workflow tests.
- [ ] Run `uv run pytest -q`.
- [ ] Run `uv run ruff check .`.
- [ ] Run `uv run basedpyright graph_memory scripts tests --level error`.
- [ ] Run `uv run python scripts/annotate_importance.py --help`.
- [ ] Run `openspec validate add-memory-stream-retrieval --strict`.
- [ ] Run `git diff --check` and review `git diff --stat`.
- [ ] Commit the forward correction and push `phase2-implement`.

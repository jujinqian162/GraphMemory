# Narrow Python Export Surfaces Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Shrink package and compatibility-module export surfaces so `__all__` contains only stable caller-facing APIs, without changing runtime behavior.

**Architecture:** Keep root workflow ports and genuinely used package entry points narrow. Remove unused package re-export facades, keep owned leaf-module imports intact, and mark compatibility projection helpers private. Architecture tests freeze the intended surfaces and direct-import tests protect dependency behavior.

**Tech Stack:** Python 3.10+, pytest, ruff, basedpyright, OpenSpec

---

### Task 1: Freeze Intended Export Surfaces

**Files:**
- Create: `tests/test_public_api_exports.py`
- Modify: `tests/test_core_refactor_batch1_boundaries.py`
- Modify: `tests/test_retrieval_registry_projections.py`
- Modify: `tests/test_phase1_real_retrieval.py`

- [x] Add assertions for narrow root/package `__all__` values and private projection helpers.
- [x] Change tests that consume implementation details through package facades to import their owned leaf modules.
- [x] Run the focused tests and confirm they fail because current export surfaces are too broad.

### Task 2: Narrow Compatibility and Package Facades

**Files:**
- Modify: `graph_memory/registry/projections.py`
- Modify: `graph_memory/registry/retrieval.py`
- Modify: `graph_memory/registry/retrieval_builders.py`
- Modify: `graph_memory/config/__init__.py`
- Modify: `graph_memory/contracts/__init__.py`
- Modify: `graph_memory/evaluation/__init__.py`
- Modify: `graph_memory/graphs/__init__.py`
- Modify: `graph_memory/graphs/construction/__init__.py`
- Modify: `graph_memory/graphs/construction/rules/__init__.py`
- Modify: `graph_memory/models/graph_retriever/config/__init__.py`
- Modify: `graph_memory/registry/__init__.py`
- Modify: `graph_memory/registry/stage_configs.py`
- Modify: `graph_memory/registry/training.py`
- Modify: `graph_memory/retrieval/execution/__init__.py`
- Modify: `graph_memory/retrieval/methods/flat/__init__.py`
- Modify: `graph_memory/retrieval/methods/graph_rerank/__init__.py`
- Modify: `graph_memory/retrieval/tuning/__init__.py`
- Modify: `graph_memory/text/__init__.py`
- Modify: `graph_memory/training_pairs/__init__.py`
- Modify: `graph_memory/io.py`
- Modify: `graph_memory/infrastructure/io.py`
- Modify: `graph_memory/training_config.py`

- [x] Rename compatibility-only helpers with a leading underscore and update internal callers.
- [x] Empty package facades that have no production consumers.
- [x] Retain only script-facing or stable domain entry points in non-empty package facades.
- [x] Run focused tests until green.

### Task 3: Verify and Commit

**Files:**
- Verify all modified source and test files.

- [x] Run the full pytest suite in an environment with a writable pytest temp directory.
- [x] Run `ruff check`, `basedpyright --outputjson --level error`, strict OpenSpec validation, and `git diff --check`.
- [x] Review `git diff` and confirm unrelated `.gitignore` and `.ckignore` changes are excluded.
- [x] Stage only this change and commit it.

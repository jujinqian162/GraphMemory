# Commands

Date: 2026-05-20

Status: Pre-implementation placeholder. Fill this document after Phase 1 scripts are implemented.

## Purpose

This document is the canonical place for the exact command sequence for running the Phase 1 experiment.

It should be updated after the CLI scripts exist so the documented commands match the real implementation.

The root `README.md` should provide a short quick-start and link here instead of duplicating the full runbook.

## Required Sections

The implemented command guide must include:

- preparing train/dev/test input and label artifacts
- building graphs
- running BM25 retrieval
- running dense retrieval
- tuning BM25 graph rerank on dev
- tuning dense graph rerank on dev
- running fixed graph rerank configs on test
- evaluating all methods
- aggregating final tables
- running leakage checks
- running tests

## Rules

- Prefer leakage-safe commands using `.input.json` and `.labels.json`.
- Include compatibility notes for the original project command surface where relevant.
- Show the expected output paths for every command.
- Keep command examples synchronized with actual script arguments.
- Link to `implementation-handoff.md` so users can move from running the system to reviewing the code.

---
name: engineering-preflight-planning
description: Use when a user wants to prepare a codebase before implementation by discussing, organizing, and documenting engineering quality, architecture, abstractions, data contracts, testing, logging, reproducibility, validation, or debug strategy.
---

# Engineering Preflight Planning

## Overview

Use this skill to turn pre-implementation discussion into durable, implementation-ready project guidance. Keep the work in planning/design space: clarify decisions, organize docs, persist outcomes, and verify documentation before coding starts.

## Core Rule

Do not jump into implementation. First make the project understandable enough that implementation can proceed with fewer hidden assumptions.

```text
requirements -> discussion -> decisions -> docs -> verification -> optional commit
```

## Workflow

### 1. Gather Context

Read the user's referenced requirement documents, phase plans, existing docs, and current repository structure.

Look for:

- fixed project structure or command requirements
- required artifacts and schemas
- phase boundaries and non-goals
- existing documentation conventions
- prior decisions already recorded in docs

If OpenSpec exists, check active changes before creating new design artifacts.

### 2. Organize Documentation First

If docs are flat or hard to navigate, propose or create a layered structure:

```text
docs/
  README.md
  00-overview/
  10-plans/
  20-contracts/
  30-design/
  40-operations/
  archive/
```

Use each layer for a distinct purpose:

| Layer | Purpose |
|---|---|
| `00-overview` | background, requirements, roadmap, project metadata |
| `10-plans` | phase plans, brainstorm logs, execution notes |
| `20-contracts` | schemas, artifact contracts, validation rules |
| `30-design` | architecture, abstractions, naming, testing, validation |
| `40-operations` | commands, logging, reproducibility, debug artifacts |
| `archive` | original source docs and superseded material |

Maintain `docs/README.md` as the navigation and maintenance guide.

### 3. Discuss Topics In A Useful Order

Use this order unless the user steers elsewhere:

1. Engineering quality principles.
2. Documentation structure and maintenance method.
3. Data contracts and artifact boundaries.
4. Architecture and abstraction boundaries.
5. Core data representation.
6. Core behavior interfaces.
7. Naming conventions.
8. Logging and run records.
9. Testing strategy.
10. Reproducibility strategy.
11. Validation strategy.
12. Debug artifact formats.
13. Remaining deferred decisions.

For each topic:

- summarize the current state
- list the important options
- recommend one path
- explain implications for testing, observability, and extensibility
- ask for confirmation when a decision changes implementation shape
- persist accepted decisions immediately

### 4. Prefer Lightweight Abstractions

Recommend abstractions only where they protect variation or testability.

```text
stable domain data     -> aliases, TypedDicts, frozen dataclasses
replaceable behavior   -> small Protocol or class
deterministic utility  -> plain function
```

Prefer:

- library-core with thin CLI
- artifacts as external contracts
- explicit validators
- run summaries for reproducibility
- small debug artifacts

Avoid:

- deep package hierarchies before the project runs
- plugin registries too early
- global config singletons
- silent fallback behavior
- large object graphs mirroring JSON

### 5. Persist Decisions As They Stabilize

Use brainstorm files for live discussion and promote stable conclusions into durable docs.

Examples:

| Discussion | Durable home |
|---|---|
| data schemas | `docs/20-contracts/phase1-data-contracts.md` |
| architecture | `docs/30-design/architecture.md` |
| abstractions | `docs/30-design/abstractions.md` |
| naming | `docs/30-design/naming-conventions.md` |
| testing | `docs/30-design/testing-strategy.md` |
| validation | `docs/30-design/validation-strategy.md` |
| logging | `docs/40-operations/logging.md` |
| reproducibility | `docs/40-operations/reproducibility.md` |
| debug artifacts | `docs/40-operations/debug-artifacts.md` |

When moving or renaming docs, update links and preserve source material in `archive/`.

### 6. Keep The User In The Loop

Use short progress updates while exploring or editing docs.

When the user asks to discuss a topic, do not stop at a single narrow point if they raised several. Address all mentioned points, identify what is agreed, and record what remains deferred.

When a concept is unclear, explain it in one small example before asking for a decision.

### 7. Verify Before Claiming Completion

Before claiming the preflight docs are complete or committing them, run checks appropriate to the repository.

Useful checks:

```text
rg -n "TBD|TODO|Status: Working draft|Open Decisions|Open Questions|Initial Open Questions" docs README.md
rg -n "old-file-name|old/path" docs README.md
```

Also check that referenced docs paths exist, review staged file names, and ensure only intended documentation files are staged.

If the user asks for a commit:

1. stage only the intended docs
2. inspect staged file names/stat
3. commit with a docs-oriented message
4. report commit hash and any verification caveats

## Good Output Shape

Prefer concise summaries with links to the durable docs:

- what was discussed
- what decision was made
- where it was persisted
- what remains deferred
- what the next useful topic is

Do not overwhelm the user with full copied documents unless they ask to review the whole artifact.

# Graph Memory Documentation

This directory is organized by abstraction level. Read from top to bottom when onboarding; write from the most stable level downward when maintaining docs.

## Documentation Map

| Layer | Directory | Purpose |
|---|---|---|
| Overview | `00-overview/` | Stable project background, requirements, roadmap, and high-level experiment intent. |
| Plans | `10-plans/` | Time-bounded implementation plans, brainstorm records, and execution notes. |
| Contracts | `20-contracts/` | Concrete artifact schemas, data contracts, config contracts, and validation rules. |
| Design | `30-design/` | Architecture layering, abstraction boundaries, extensibility, observability design. |
| Operations | `40-operations/` | Reproduction commands, logging conventions, troubleshooting, runbooks. |
| Configs | `configs/` | One-to-one Chinese reference docs for editable JSON config files. |
| Archive | `archive/` | Original source documents and superseded materials kept for provenance. |

## Current Entry Points

- Project overview: `00-overview/project-overview.md`
- Original student experiment plan: `archive/original-student-experiment-plan.md`
- Phase 1 implementation plan: `10-plans/phase1-real-graph-memory.md`
- Phase 2 R-GCN trainable retriever plan: `10-plans/phase2-rgcn-trainable-retriever.md`
- Dense-FT implementation plan: `10-plans/dense-ft-implementation-plan.md`
- Trainable-stack current-only refactor plan: `10-plans/trainable-stack-zero-compatibility-refactor-plan.md`
- Engineering quality discussion log: `10-plans/engineering-quality-brainstorm.md`
- Architecture: `30-design/architecture.md`
- Abstractions: `30-design/abstractions.md`
- Cross-dataset refactor design: `30-design/cross-dataset-refactor-design.md`
- Architecture and abstraction discussion history: `10-plans/architecture-abstraction-brainstorm.md`
- Core package layered refactor design history: `10-plans/graph-memory-core-package-refactor-design.md`
- Contract documentation guide: `20-contracts/README.md`
- Data contracts: `20-contracts/data-contracts.md`
- Retrieval contracts: `20-contracts/retrieval-contracts.md`
- Model contracts: `20-contracts/model-contracts.md`
- Config documentation index: `configs/README.md`
- Naming conventions: `30-design/naming-conventions.md`
- Logging and run records: `40-operations/logging.md`
- Testing strategy: `30-design/testing-strategy.md`
- Reproducibility strategy: `40-operations/reproducibility.md`
- Validation strategy: `30-design/validation-strategy.md`
- Debug artifacts: `40-operations/debug-artifacts.md`
- Phase 1 command runbook: `40-operations/commands.md`
- Implementation handoff: `40-operations/implementation-handoff.md`

## Current Code Navigation

The maintained package layout is domain-owned:

```text
graph_memory/contracts/
graph_memory/datasets/
graph_memory/evaluation/
graph_memory/graphs/
graph_memory/infrastructure/
graph_memory/models/dense_finetune/
graph_memory/models/graph_retriever/
graph_memory/registry/
graph_memory/retrieval/
graph_memory/stages/
graph_memory/text/
graph_memory/training_pairs/
graph_memory/validation/
```

The maintained runtime boundary is one-way:

```text
scripts/workflow/
  -> low-level scripts with complete stage configs
  -> graph_memory/stages/
  -> registry builders and domain packages
```

`graph_memory/registry/` owns current method identity, lifecycle, dependency, config, and builder dispatch. `graph_memory/stages/` owns stage execution. Root `graph_memory/io.py` and `graph_memory/observability.py` remain thin integration ports. The retired `graph_memory/retrieval_registry.py`, `graph_memory/training_config.py`, registry projection/catalog facades, and trainable compatibility loaders no longer exist.

## Maintenance Rules

### Phase Separation Policy

- Phase-specific documents are allowed in `10-plans/` because plans record time-bounded execution scope, acceptance criteria, and implementation history.
- Non-plan layers should be integrated project-level references. `00-overview/`, `20-contracts/`, `30-design/`, and `40-operations/` should evolve to cover the currently supported phases instead of creating parallel `phase1-*` and `phase2-*` documents.
- When a non-plan document needs phase-specific details, keep them as clearly labeled sections inside the same maintained document.
- If an existing non-plan document was originally phase-specific, update or rename it when the next phase depends on the same topic, then update all links.

### Choose The Right Layer

- Put stable background, goals, and cross-phase requirements in `00-overview/`.
- Put current work plans and discussion records in `10-plans/`.
- Put exact schemas, artifact contracts, and validation rules in `20-contracts/`.
- Put architecture decisions, interface boundaries, and extension strategy in `30-design/`.
- Put commands, logs, debugging, and repeatable run procedures in `40-operations/`.
- Put raw imported documents, old drafts, and superseded plans in `archive/`.

### Keep High-Level Docs Clean

High-level docs should explain intent and decisions, not implementation minutiae. If a section starts listing JSON fields, CLI flags, edge weight formulas, or validation rules, it probably belongs in `20-contracts/`, `30-design/`, or `40-operations/`.

### Prefer Stable File Names

Use stable semantic names for durable docs:

```text
data-contracts.md
retrieval-contracts.md
model-contracts.md
architecture.md
observability.md
commands.md
```

Use phase names for plan documents. Use dates only for discussion logs, meeting notes, or one-off historical records. Once a document becomes a maintained non-plan reference, remove `draft`, date noise, and phase-specific naming unless the document is being kept as historical plan material.

### Track Provenance Without Polluting The Main Path

Keep original source material in `archive/` instead of rewriting it in place. When a stable document is derived from an archived source, link back to the source and summarize only the durable decisions.

### Keep Contracts Strict

Contract documents should include:

- Producer and consumer.
- Required fields.
- Forbidden fields.
- Field meaning.
- Invariants.
- Minimal examples.
- Validation and failure behavior.

### Update Links When Moving Docs

Before moving or renaming a document, search for references:

```powershell
rg "old-file-name|old/path" docs README.md
```

After moving, update all links and run the search again until no stale references remain.

### Promote Discussion Into Design

Brainstorm records are allowed to be messy while decisions are forming. Once a decision stabilizes, promote it into the appropriate durable document and leave the discussion log as historical context.

Examples:

- Data schema discussion -> `20-contracts/data-contracts.md`
- Abstraction discussion -> `30-design/abstractions.md`
- Logging discussion -> `40-operations/logging.md`

### Do Not Duplicate Truth

Each durable decision should have one canonical home. Other documents should link to it instead of copying the same rule. This matters for experiment settings, schema fields, metric definitions, and command sequences.

### Fail-Fast Documentation Style

For experiment code, documentation should prefer explicit constraints over vague guidance. If violating a rule would invalidate results, write it as a contract or invariant, not as a suggestion.

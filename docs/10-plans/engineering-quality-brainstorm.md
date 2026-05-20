# Engineering Quality Brainstorm

Date: 2026-05-20

Status: Draft discussion record. This file captures decisions while we prepare the Phase 1 codebase. It is not yet the final OpenSpec artifact.

## Current Decisions

### Code Style

- Type annotations should be precise but not complex.
- Prefer Java-like readability: clear named types, small domain objects, and explicit function signatures.
- Avoid unreadable inline nested types such as deeply nested `Tuple[List[int], ...]`.
- If a type becomes hard to read inline, introduce a named dataclass, `TypedDict`, type alias, enum, or protocol.
- Code should optimize for experiment correctness and reviewability over cleverness.
- Public functions should expose clear domain-level types.
- Complex return values should use named result objects instead of raw tuples.
- Internal local variables do not need excessive type annotations when the meaning is already obvious.
- Naming conventions are defined in `docs/30-design/naming-conventions.md`.

### Project Structure

- No additional discussion needed for now.
- Follow the project requirement document and the Phase 1 plan.

### Data Contracts

- Data contracts need explicit documentation.
- The schemas for task inputs, labels, graphs, ranked results, configs, and metrics should be documented before implementation depends on them.
- Data contracts should become the main agreement between independent modules.
- Data schema documentation should include required fields, forbidden fields, field meaning, and example records.
- The implementation should validate important contract assumptions and fail fast when records violate the contract.
- Draft contract document created at `docs/20-contracts/phase1-data-contracts.md`.
- The contract document should focus on Phase 1 artifacts first and avoid pulling in Phase 2/3 complexity early.

### Leakage Boundary

- Current position: do not over-engineer this area.
- Retrieval and graph-construction code should simply not read label-only fields.
- Any future contract should still make input artifacts and label artifacts clear enough that accidental leakage is hard to introduce.
- Leakage prevention should primarily be achieved by clear artifact separation and module inputs, not by complicated runtime machinery.
- Tests can still assert that input and graph artifacts do not contain label-only fields.

### Configuration

- Priority rule: CLI arguments override config file values.
- Config files define reproducible defaults.
- CLI exists for temporary experimental overrides.
- Config loading should be explicit and deterministic.
- The final effective configuration should be easy to inspect in logs or run metadata.

### Error Handling

- Prefer fail-fast behavior.
- If data, schema, config, split, or metric assumptions are violated, raise an exception and stop.
- Avoid fallback behavior that silently changes the experiment or produces inaccurate results.
- Fallbacks should be reserved for non-scientific convenience only, and even then they should be explicit.
- Missing files, malformed schemas, overlapping splits, unsupported methods, and impossible metric states should terminate the run.

### Performance And Cache

- Defer performance and cache design.
- Do not introduce caching complexity during initial Phase 1 construction unless a concrete bottleneck appears.
- Prefer simpler, directly inspectable runs over premature caching.
- If caching is added later, it must be keyed by dataset, config, model, and code-relevant parameters so stale results are not reused.

### Documentation

- Main project documentation should be advanced through OpenSpec.
- This brainstorm file is a temporary discussion log that can later feed OpenSpec/design artifacts.
- Documentation should focus on contracts, commands, architecture decisions, and experiment interpretation rather than decorative prose.
- OpenSpec should be the durable source for formalized requirements and design.
- Documentation is organized by abstraction level under `docs/README.md`.
- High-level overview, plans, contracts, design, operations, and archive material should not be mixed in one flat directory.

### Workflow

- Follow Superpowers guidance for brainstorming, planning, testing, and implementation.
- Use brainstorming before major design changes.
- Use planning before multi-step implementation.
- Use test-driven or verification-first habits where correctness is experimentally important.
- Testing strategy is defined in `docs/30-design/testing-strategy.md`.

### Logging

- Logging and run records are defined in `docs/40-operations/logging.md`.
- Console logs should be human-readable and stage-based.
- Run summaries should capture effective config, input/output paths, counts, timings, and environment notes.
- Debug artifacts should be optional and bounded.
- Retrieval and graph-construction logs/debug outputs must not include label-only fields.

### Reproducibility

- Reproducibility strategy is defined in `docs/40-operations/reproducibility.md`.
- Config files define repeatable defaults.
- CLI arguments may override config for temporary runs.
- Every script must write a run summary.
- Dataset splits must be deterministic and documented.
- Dev tuning and test evaluation must stay separate.
- Command usage documentation must be written after implementation in `docs/40-operations/commands.md`.

### Validation

- Validation strategy is defined in `docs/30-design/validation-strategy.md`.
- Validators should run at script boundaries.
- Validators should raise clear exceptions and never silently repair artifacts.
- Unknown top-level fields should fail unless placed under explicit `metadata` or `debug` objects.
- Label leakage validation is required for input-visible artifacts.

### Debug Artifacts

- Debug artifact formats are defined in `docs/40-operations/debug-artifacts.md`.
- Per-task debug records should use JSONL.
- Aggregate debug summaries should use JSON.
- Debug artifacts should be optional, bounded, and separate from canonical result artifacts.
- Retrieval-stage debug artifacts must not contain gold labels.

## Discussion Notes

### Type Annotation Policy

The desired style is realistic in Python. The key is to avoid making the type system carry too much structure inline. Complex structures should be named so signatures read like domain language.

Preferred examples:

- `TaskId = str`
- `NodeId = str`
- `MemoryTaskInput`
- `MemoryTaskLabels`
- `MemoryGraph`
- `RankedResult`

Avoid signatures that expose deeply nested containers. When a return value has more than one meaningful component, prefer a named dataclass over a tuple.

### Data Contract Documentation

Data contracts are central because this project has many file boundaries: task input, labels, graph, retrieval output, tuning config, evaluation table. The code should not rely on implicit knowledge of JSON fields.

Contract docs should answer:

- Which module writes this artifact?
- Which module reads it?
- Which fields are required?
- Which fields are forbidden?
- Which fields are label-only?
- What invariants must hold?
- What is a minimal valid example?

### Leakage Design

The current direction is intentionally lightweight. We do not need a complicated access-control layer. The stronger engineering move is to make module inputs honest: graph construction and retrieval receive input records only; evaluation and tuning receive labels.

This is still worth documenting because leakage in experiment code often enters through convenience shortcuts, not malicious logic.

### Configuration Priority

The selected rule is:

```text
effective_config = default_config + config_file + cli_overrides
```

where CLI overrides win. This supports reproducibility through config files while preserving quick experimental changes through CLI flags.

### Error Handling Philosophy

Fail fast is the right default for this project. Silent recovery can invalidate an experiment more dangerously than a crash. A failed run is visible; a subtly wrong CSV is expensive.

Examples that should raise:

- Requested split exceeds available labeled data.
- Dev and test split overlap.
- A supporting fact cannot map to a memory node.
- A graph references a missing node.
- A prediction includes duplicate ranked node ids.
- A metric is asked to evaluate a missing task id.
- A graph-rerank method is run without graphs or graph config.

### Deferred Performance And Cache

Performance and caching are intentionally out of scope for initial construction. They can hide data freshness issues and complicate debugging. The first version should be simple, deterministic, and inspectable.

### Observability

Observability is intentionally lightweight. It means small, explicit debug surfaces rather than a monitoring system.

Current low-complexity observability surfaces:

- A per-run summary JSON with effective config, input sizes, method, timing, and artifact paths.
- Optional per-task debug dumps for a small task subset.
- Graph statistics: node count, edge count by type, average degree, isolated nodes.
- Retrieval diagnostics: top ranked nodes with score components for graph rerank.
- Leakage check command or validation report.
- Evaluation failure cases: tasks where full support failed, graph connected gold evidence, or dense retrieved only one hop.

The current observability foundation is mandatory `run_summary.json`, graph stats, bounded debug artifacts on request, and evaluation failure-case exports. This keeps CLI overrides, effective config, counts, timings, and paths inspectable without adding a monitoring framework.

## Deferred To Implementation

- Dependency and environment policy.
- CLI and configuration contract details.
- Implementation readiness checklist.
- Dependency and environment details depend on the actual package setup and local model availability.
- CLI parameter details are intentionally deferred; command usage documentation must be written after implementation in `docs/40-operations/commands.md`.

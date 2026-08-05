# Implementation verification

## Cutover state

- Retired `graph_memory/validation/` in full.
- Retired `graph_memory/contracts/ranking.py`, `graphs.py`, `training_pairs.py`, `metrics.py`, and `errors.py` without forwarding modules.
- Domain models reject unknown fields, use authoritative enums/literals, and serialize through Pydantic JSON mode.
- Retrieval validates each task result before invoking the next task.
- Dense-FT, evidence R-GCN, and provenance R-GCN payload aggregates validate before model/encoder/optimizer work.
- Framework tensor shape/device/index assertions remain in tensor/runtime code.

## Verification commands

Executed successfully on the implementation checkout:

```text
uv run pytest -q
uv run ruff check graph_memory experiment scripts tests
uv run basedpyright --level error
python -m compileall -q graph_memory scripts tests
git diff --check
openspec validate unify-pydantic-domain-validation --strict
```

The final pytest collection contains 193 tests (baseline: 187); the added cases cover shared Pydantic mechanics, enum-authoritative retrieval methods, per-task early abort, fail-fast training payloads, and retired-surface architecture guards.

## Representative parse timing

A warmed 500-iteration local microbenchmark using module-level `TypeAdapter`s measured:

| Payload | Records | Mean parse time |
|---|---:|---:|
| 2Wiki-provenance v3 raw record | 1 | 1.171 ms |
| EvidenceGraph batch | 1 | 0.071 ms |
| TrainPairRecord batch | 1 | 0.001 ms |
| RankedResult batch | 1 | 0.005 ms |

These are startup/boundary measurements, not performance guarantees. Adapter/model validation occurs while stages read or construct artifacts and before DataLoader creation. No Pydantic parse was added to `DataLoader.__getitem__`, graph collation, optimizer steps, or epoch loops.

## Residual scans

The cutover scan covers production, scripts, and tests for:

- `graph_memory.validation` imports;
- retired `graph_memory.contracts.{ranking,graphs,training_pairs,metrics,errors}` imports;
- `ContractValidationError` and retired validator calls;
- reintroduced retired paths through architecture tests.

The validator-to-owner mapping is recorded in `migration-matrix.md`.

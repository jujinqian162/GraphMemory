## 1. Freeze the Contract and Build the Migration Matrix

- [x] 1.1 Inventory every public function, private scientific helper, field-set constant, allowlist, caller, test, and error family in `graph_memory/validation/`, and map each one to an owning Pydantic model/aggregate and focused test.
- [x] 1.2 Inventory parallel validation outside the package, including provenance-RGCN checkpoint payload checks, graph/provenance dataclass `__post_init__` checks, source/project artifact parsers, trace serializers, metric column lists, and direct JSON casts.
- [x] 1.3 Record normalized JSON round-trip fixtures for current HotpotQA, 2Wiki, MuSiQue, 2Wiki-provenance v3, EvidenceGraph, train-pair, ranked-result/native-trace, metric, Dense-FT metadata, evidence checkpoint, and provenance checkpoint schemas.
- [x] 1.4 Add a compact parameterized behavior-freeze matrix covering every custom invariant family currently enforced by ranking, graph, dataset, provenance, pair, model/checkpoint, metric, leakage, and join validators.
- [x] 1.5 Add the historical regression proving every active `RetrievalMethodId` is accepted by the result contract without a copied allowlist.
- [x] 1.6 Record the current focused/full verification commands, test count budget, representative artifact parse timings, and active trace/checkpoint/schema inventory before production migration.

## 2. Establish Pydantic Primitives and Correct Ownership

- [x] 2.1 Add the minimal low-level frozen/closed `DomainModel` base plus strict integer/boolean, finite float, non-negative/positive scalar, non-empty ID, JSON metadata, and reusable label-free JSON types.
- [x] 2.2 Add representative tests for strict scientific scalar behavior, unknown-field rejection, immutable models, JSON tuple/list round trips, enum/literal dumping, aliases, optional-field omission, and finite-number rejection without exhaustively retesting Pydantic.
- [x] 2.3 Move `RetrievalMethodId` to a retrieval-owned low-level module and make Registry/configuration import or re-export that single enum without changing public string values.
- [x] 2.4 Move graph node/edge/provenance enums and train-pair sample enums to their owning low-level domains where current generic-contract ownership would create upward imports or duplicated allowlists.
- [x] 2.5 Update architecture dependency tests to allow the new low-level model base while prohibiting experiment/stage/Registry imports from domain models.
- [x] 2.6 Prove Hydra composition, Prefect parameter serialization, and current Python 3.10-compatible Pydantic behavior remain valid with the shared domain models.

## 3. Migrate Retrieval Requests, Results, and Native Traces

- [x] 3.1 Convert `TextCandidate`, text/graph/GraphRAG/execution-provenance ranking requests, and `RetrievalExecutionTask` from validation-bearing dataclasses to closed Pydantic models with task/candidate/graph consistency invariants.
- [x] 3.2 Replace `RankedNodeRecord`, `RetrievedSubgraph`, result metadata, and `RankedResult` with retrieval-owned Pydantic models preserving the exact current JSON schema.
- [x] 3.3 Convert every currently accepted native trace kind and nested trace record to a closed Pydantic discriminated union, reconciling the inventory with any concurrently landed method change before deleting a trace branch.
- [x] 3.4 Move all entity-search, local bridge, execution-provenance, stateless/local provenance, and query-conditioned provenance custom invariants from `validation/ranking.py` into their owning trace/member models.
- [x] 3.5 Replace the hand-written `_native_trace_record`/`isinstance` serializer chain with model JSON dumps and prove exact valid-fixture round trips including omitted optional fields.
- [x] 3.6 Add a per-task request/result aggregate covering task identity, candidate uniqueness and exact coverage, descending finite scores, subgraph node/edge references, metadata policy, and trace candidate/native-graph context.
- [x] 3.7 Change retrieval execution to construct and validate one result immediately after each method call, then run only a cheap final batch task-coverage/uniqueness aggregate.
- [x] 3.8 Add a fake-method regression proving an invalid first result prevents execution of the second task.
- [x] 3.9 Route evidence R-GCN dev prediction assembly through the production result factory and per-task aggregate before metric calculation or checkpoint selection.
- [x] 3.10 Route provenance R-GCN dev prediction/structured-trace assembly through the same production result contract where that trainer emits ranked results.
- [x] 3.11 Update retrieval methods, execution services, stages, dataset evaluation projectors, graph views, path metrics, benchmarks, scripts, and tests from dictionary indexing/TypedDict annotations to model APIs.
- [x] 3.12 Delete `graph_memory/validation/ranking.py` and `graph_memory/contracts/ranking.py` only after all ranked-result and native-trace freeze cases pass with no production fallback.

## 4. Migrate Evidence and Execution-Provenance Graph Contracts

- [x] 4.1 Replace question/item node, graph edge, and EvidenceGraph TypedDicts with discriminated closed Pydantic models preserving current field names and JSON output.
- [x] 4.2 Move one-query-node, unique-ID, node-shape, endpoint, edge-type, directed, finite-weight, request-item coverage, task alignment, and leakage invariants into graph models and graph/request aggregates.
- [x] 4.3 Convert `FieldBinding`, execution-provenance node/edge/graph values from validation-bearing dataclasses to closed Pydantic models with the full legal typed-transition matrix.
- [x] 4.4 Make 2Wiki-provenance record graphs embed/use the same execution-provenance graph model as retrieval requests and remove record-to-runtime graph reconstruction/duplicate validation.
- [x] 4.5 Update graph construction, statistics, views, batching, tensorization, projectors, retrieval requests, pair samplers, evaluation, stages, scripts, fixtures, and tests to consume graph model attributes.
- [x] 4.6 Add graph artifact/request batch adapters at build, publication, and consumption boundaries and prove exact active JSON round trips.
- [x] 4.7 Delete `graph_memory/validation/graphs.py` and retire `graph_memory/contracts/graphs.py` artifact shapes after all graph behavior and architecture tests pass.

## 5. Migrate Dataset Records and Prepared-Split Joins

- [x] 5.1 Convert HotpotQA candidate, ranking, label, combined inspection, and prepared-split records to dataset-owned closed Pydantic models.
- [x] 5.2 Convert 2Wiki candidate, ranking, label, combined inspection, and prepared-split records to dataset-owned closed Pydantic models.
- [x] 5.3 Convert MuSiQue candidate, ranking, label, combined inspection, and prepared-split records to dataset-owned closed Pydantic models.
- [x] 5.4 Convert 2Wiki-provenance candidate, binding, confidence metadata, node/edge/graph, ranking, label, raw, combined, and prepared-split records to dataset-owned closed Pydantic models using the authoritative provenance graph type.
- [x] 5.5 Move dataset prefix, candidate ID/position, required metadata, gold membership, dependency endpoint/order, ranking-label task alignment, and no-label-leakage invariants into record/split models.
- [x] 5.6 Move every current provenance-v3 construction identity, fixed degree, branch role/bucket, confidence range/mass, binding, returns/feed pairing, candidate-output coverage, and gold path invariant into 2Wiki-provenance models/aggregates.
- [x] 5.7 Change raw-source parsers and converters to return models at the first project-owned artifact boundary while keeping third-party source-format parsing errors explicit and dataset-owned.
- [x] 5.8 Update prepare/transform stages, selection/projector dispatch, combined-record builders, scripts, fixtures, and tests to pass model instances and dump JSON only at publication.
- [x] 5.9 Prove current valid prepared artifacts preserve normalized JSON and every invalid dataset/provenance freeze case remains rejected for the same scientific reason.
- [x] 5.10 Delete `graph_memory/validation/tasks.py` and `graph_memory/validation/twowiki_provenance.py` after all dataset callers have migrated.

## 6. Migrate Train-Pair Contracts and Configuration

- [x] 6.1 Replace TrainPairRecord, TrainPairBuildSummary, NegativeSamplingConfig, ProvenanceNegativeSamplingConfig, and pair-build task/result dataclasses with training-pair-owned Pydantic models where they define validation or serialization.
- [x] 6.2 Encode label/sample-type consistency, candidate/optional-graph membership, query exclusion, duplicate keys, exact positive/gold equality, negative/gold exclusion, and task coverage in a typed pair-dataset aggregate.
- [x] 6.3 Encode summary count/range/config/no-positive/overlap/shortfall/source-precedence invariants in the pair result/summary models and derive sample allowlists from the authoritative enum.
- [x] 6.4 Make experiment pair configuration compose/reuse domain pair configuration and remove stage conversion through equivalent dataclasses and `asdict`/`**model_dump()` copies.
- [x] 6.5 Update generic and provenance pair builders/samplers, stages, trainers, artifact publishers/readers, MLflow projections, tests, and fixtures to consume pair models.
- [x] 6.6 Validate pair aggregates before any Dense-FT/evidence-RGCN/provenance-RGCN model or encoder initialization and add explicit fail-before-training tests.
- [x] 6.7 Prove pair and summary JSON output, deterministic ordering, provenance overlap accounting, and ablation/cache identity remain unchanged for valid inputs.
- [x] 6.8 Delete `graph_memory/validation/training_pairs.py` and retire `graph_memory/contracts/training_pairs.py` artifact shapes after migration.

## 7. Unify Training Configuration, Preflight, and Checkpoints

- [x] 7.1 Convert graph-retriever NodeFeatureConfig, RgcnModelConfig, RgcnTrainingConfig, selection/inference configuration, and other validation-bearing model config records to authoritative low-level Pydantic models.
- [x] 7.2 Make experiment method/stage models, Registry settings/build payloads, trainer adapters, tracking projections, and model factories reuse the authoritative graph-retriever config models without copied fields/defaults or `to_json_dict` methods.
- [x] 7.3 Preserve all feature/relation vocabulary, optimizer, batch-size, dropout, graph encoder, message transform, edge-weight, enabled-edge, ablation, loss, inference, and selection constraints in model validators.
- [x] 7.4 Convert Dense-FT run/metadata/selection records that currently rely on dataclass TypeAdapter validation to closed Pydantic models and preserve metadata JSON.
- [x] 7.5 Introduce a closed evidence R-GCN checkpoint envelope with typed metadata/config and opaque required model/optimizer/scheduler state maps; use it for both save and load.
- [x] 7.6 Introduce a closed provenance R-GCN checkpoint envelope covering its active schema, variant, construction, pair, loss, weight, inference, and selection identities plus opaque state maps; remove the parallel `_validate_payload` implementation.
- [x] 7.7 Add typed checkpoint load requests/envelopes that enforce expected method/family/variant without passing an untyped expected string into a free validator.
- [x] 7.8 Build complete dense/evidence/provenance training preflight aggregates and run them before encoder/model loading, corpus encoding, tensor materialization, optimizer creation, or epoch zero.
- [x] 7.9 Ensure dev ranked results and dev metric rows validate before selection-policy calculation, best-state replacement, training-history append, checkpoint callbacks, or final model publication.
- [x] 7.10 Add focused tests proving invalid config/input fails before expensive factory calls, invalid dev output cannot select a checkpoint, valid checkpoint round trips preserve state objects, and framework tensor-key errors remain framework-owned.
- [x] 7.11 Delete `graph_memory/validation/model.py` and obsolete graph-retriever config dataclasses/manual JSON conversion after checkpoint and preflight parity passes.

## 8. Migrate Evaluation and Metric Contracts

- [x] 8.1 Convert EvidenceLabel, evidence evaluation input, aggregate metric row, per-task row, failure case, and table row contracts to evaluation-owned Pydantic models/aggregates.
- [x] 8.2 Represent current human-readable metric column names as Pydantic aliases and derive main/path/efficiency/wide projections from the suite contract rather than duplicated validation column constants.
- [x] 8.3 Move task/label/graph/prediction alignment, one-method-per-file, gold-node membership, schema discriminator, finite/range/`N/A`, latency, edge/path, and mixed-schema invariants into evaluation/suite models.
- [x] 8.4 Change the metric suite protocol to expose its row model/adapter and make evaluate/report/analysis/delivery paths accept typed rows before CSV/JSONL dumping.
- [x] 8.5 Update checkpoint selection and training evaluation consumers to use model attributes/declared aliases without converting rows to unvalidated dictionaries.
- [x] 8.6 Prove current evidence/provenance formulas, column names/order, CSV/JSONL content, failure cases, per-task rows, and report tables remain unchanged for valid inputs.
- [x] 8.7 Delete `graph_memory/validation/metrics.py` and retire duplicated `contracts/metrics.py` artifact shapes after suite/table parity passes.

## 9. Convert Every Scientific Artifact IO Boundary

- [x] 9.1 Add module-level cached TypeAdapters for each dataset split, graph batch, pair artifact, prediction batch, checkpoint envelope, metric row/table, and other migrated scientific payload.
- [x] 9.2 Replace unchecked `read_json` casts in prepare/transform, graphs, pairs, models, retrieve, evaluate, benchmark, experiment task, and analysis paths with immediate owning-model validation.
- [x] 9.3 Replace unchecked JSONL/CSV scientific row parsing with suite/domain adapters where maintained code consumes those artifacts.
- [x] 9.4 Change every migrated publisher to serialize validated models explicitly in JSON mode with aliases and the correct optional-field policy before hashing/writing.
- [x] 9.5 Remove model-to-dict-to-model and dataclass-to-dict conversion cycles inside stages; pass frozen model instances through projectors, builders, trainers, retrieval, and evaluation.
- [x] 9.6 Verify Prefect task serialization, processed artifact references, cache identities, MLflow parameter/metric logging, output delivery, and deterministic hashes with model inputs/dumps.
- [x] 9.7 Benchmark representative full prepared, provenance, graph, pair, and prediction artifact parsing and prove validation is outside DataLoader `__getitem__`, collation, and epoch hot loops.

## 10. Remove the Legacy Validation and Error Surfaces

- [x] 10.1 Move the remaining `validate_task_id_alignment`, label-free recursion, strict scalar, unique-ID, and record-list/map responsibilities from `validation/common.py` into typed local/aggregate models or reusable annotated primitives.
- [x] 10.2 Replace maintained `ContractValidationError` expectations with Pydantic `ValidationError` location/reason assertions for model failures and owning-domain errors for computation preconditions.
- [x] 10.3 Remove `graph_memory/contracts/errors.py` if no owning non-validation domain still requires it; do not retain it as a Pydantic wrapper.
- [x] 10.4 Delete `graph_memory/validation/__init__.py`, `graph_memory/validation/common.py`, and the now-empty `graph_memory/validation/` directory.
- [x] 10.5 Remove all imports/re-exports of retired validators, `ContractValidationError`, and ranking/graph/pair/metric TypedDict artifacts from production code, tests, scripts, and maintained docs.
- [x] 10.6 Remove copied allowed-field constants, independent method/sample/edge allowlists, `_required_*` dictionary readers, hand trace dispatch/serializers, redundant `to_json_dict`, and dead conversion helpers identified in the migration matrix.
- [x] 10.7 Add architecture/import guards proving the retired package/module cannot return and low-level models do not depend upward on Registry, stages, experiment, or scripts.
- [x] 10.8 Add an AST guard prohibiting direct casts of decoded scientific artifact payloads and a focused guard against recreated parallel validation-only allowlists.
- [x] 10.9 Update `tests/test_architecture_invariants.py` package roots and maintained extension guidance to reflect domain-owned Pydantic contracts and retained tensor/runtime assertions.

## 11. Documentation and Concurrent-Change Reconciliation

- [x] 11.1 Update `docs/30-design/architecture.md` to remove `validation/`, show the new model ownership map, and document typed artifact read/write and fail-fast boundaries.
- [x] 11.2 Update maintained contract/operations/runbook documentation that tells contributors to call `validate_*` or edit `validation/ranking.py`, including retrieval trace and 2Wiki-provenance guidance.
- [x] 11.3 Reconcile the final active trace, provenance-v3, pair, metric, and checkpoint invariant inventory with changes merged after this proposal; record any explicit algorithm-owned retirement rather than silently dropping its model.
- [x] 11.4 Document that historical OpenSpec changes centralizing validators are superseded by this change while leaving archived/historical records intact.
- [x] 11.5 Document the contributor rule: add or modify the owning model and focused custom-invariant test, never a parallel field set, serializer, or central validator.

## 12. Final Verification and Cutover

- [x] 12.1 Run focused dataset/provenance, graph, pair, retrieval/trace, dev-evaluation, checkpoint, metric/table, artifact, Registry, Prefect workflow, and architecture tests after each vertical migration.
- [x] 12.2 Run the compact full pytest suite outside the Windows filesystem sandbox and keep the repository's declared collection budget.
- [x] 12.3 Run `uv run ruff check` and `uv run basedpyright --level error` outside the Windows filesystem sandbox.
- [x] 12.4 Run `python -m compileall graph_memory scripts tests`, `git diff --check`, and strict validation for this change plus `openspec validate --all --strict --no-interactive`.
- [x] 12.5 Run residual scans confirming no `graph_memory.validation`, `graph_memory.contracts.ranking`, retired validator, hand trace dispatch, duplicate validation field set, unchecked scientific IO cast, or compatibility facade remains.
- [x] 12.6 Regenerate representative current-schema artifacts and compare normalized JSON, deterministic ordering/content, active schema/method/trace identities, metric tables, and cache inputs with the frozen baseline.
- [x] 12.7 Record representative preflight/parse timings and prove no schema validation was introduced inside training epoch/DataLoader hot paths.
- [x] 12.8 Review the completed migration matrix row by row and block completion until every old custom invariant has an owning model, caller migration, and focused behavioral test.

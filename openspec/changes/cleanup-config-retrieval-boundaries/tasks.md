## 1. Config and Conversion Boundaries

- [x] 1.1 Add failing tests for optional profile selectors and no-profile stage declarations.
- [x] 1.2 Implement optional `StageConfigSpec.profile_name` and tighten config loader namespace typing.
- [x] 1.3 Add failing tests for centralized dense/trainer settings conversion adapters.
- [x] 1.4 Implement shared conversion adapters and replace script/dataclass-local conversion logic.
- [x] 1.5 Move legacy training config loading helpers to an explicit config compatibility boundary and update imports/tests.

## 2. Retrieval Build Payloads and Checkpoint Dependencies

- [x] 2.1 Add failing tests for `require_payload()` and method-specific retrieval build payloads.
- [x] 2.2 Replace `RetrievalDependencies` with object payload input plus builder-local typed payload validation.
- [x] 2.3 Add failing tests proving checkpoint loader does not create default seed/text providers.
- [x] 2.4 Move checkpoint provider assembly into the registry checkpoint builder.

## 3. Training Labels and Retrieval Result Protocol

- [x] 3.1 Add failing tests for optional `--train_labels` and train-label validation when provided.
- [x] 3.2 Thread optional train labels through the train script, stage, registry trainer, and model training call.
- [x] 3.3 Add failing tests for structured retrieval method results and seed-ranker naming.
- [x] 3.4 Implement `RetrievalMethodResult`/trace return objects and migrate production code to `SeedRanker`.

## 4. Verification

- [x] 4.1 Run focused tests for config loader, registry stage configs, retrieval builders, retrieval execution, and R-GCN training.
- [x] 4.2 Run full verification: pytest, ruff, basedpyright, openspec validate, and git diff check.

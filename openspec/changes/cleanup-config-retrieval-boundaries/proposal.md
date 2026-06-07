## Why

The config-registry refactor left several compatibility seams in production code: profile hooks leak into stage declarations, settings-to-runtime conversions are scattered, retrieval builders accept a broad optional dependency bag, and checkpoint inference still creates default dependencies internally. This change tightens those boundaries without changing the public workflow semantics.

## What Changes

- Make profile resolution an optional config-loader concern, so stages without profile support do not declare `_no_profile`.
- Centralize conversions from registry settings into runtime configs.
- Replace the broad retrieval dependency bag with object payload input plus builder-local `require_payload()` validation into concrete build payloads.
- Move checkpoint graph retriever dependency assembly out of the checkpoint loader and into the registry builder layer.
- Make train labels optional at the CLI boundary while still using them for train-pair validation when provided.
- Clarify retrieval protocols by separating full seed ranking from final retrieval methods and by returning a structured retrieval result.
- Mark legacy training config loading helpers as compatibility-only and keep production entrypoints on `CONFIG_LOADER.load(Registry.configs.*)`.

## Capabilities

### New Capabilities
- `config-retrieval-boundary-cleanup`: Covers config/profile application boundaries, registry-to-runtime conversion adapters, retrieval build payload validation, checkpoint dependency assembly, train-label validation flow, and structured retrieval method results.

### Modified Capabilities

## Impact

- Affects config loader/spec declarations, registry settings and retrieval builders, retrieval contracts/execution methods, train stage/script plumbing, and tests for the config-registry refactor.
- External workflow commands should remain compatible except `scripts/train_graph_retriever.py --train_labels` becomes optional.

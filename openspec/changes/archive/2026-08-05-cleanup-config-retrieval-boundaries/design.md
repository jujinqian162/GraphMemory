## Context

The current config-registry implementation already routes script entrypoints through `CONFIG_LOADER.load(Registry.configs.*)`, but several details still blur responsibility boundaries. Stage specs expose no-op profile hooks, runtime config conversions are implemented as local private helpers, retrieval builders accept one broad optional dependency structure, and checkpoint graph inference creates default seed/text providers inside the low-level loader.

The intended boundary is: scripts and stages load artifacts and orchestrate work; config loading applies profiles and CLI patches before domain code sees the config; registry builders construct runtime methods from typed settings and runtime payloads; retrieval/model implementations operate on already-assembled dependencies.

## Goals / Non-Goals

**Goals:**
- Keep profile application inside config loading and remove no-op profile declarations from normal stage specs.
- Centralize settings-to-runtime config conversion helpers.
- Replace the broad retrieval dependency bag with builder-local typed payload validation.
- Move checkpoint graph retriever dependency assembly up to the registry builder layer.
- Preserve existing retrieval/training behavior while making train labels optional at the CLI boundary.
- Make retrieval method results explicit instead of returning tuple payloads.

**Non-Goals:**
- Do not change workflow artifact semantics or canonical script entrypoints.
- Do not move artifact file reading into retrieval method implementations.
- Do not change default training/inference device selection from `cpu`.
- Do not remove `checkpoint_callback` in this change.

## Decisions

1. **Profile handling stays in config loading.** `StageConfigSpec.profile_name` becomes optional, and `ConfigLoader` only applies profiles when a spec explicitly provides a selector. This avoids leaking profile semantics into stages that only consume effective configs.

2. **Conversions become named adapter functions.** Registry settings remain pure data. Functions such as `dense_config_from_encoder_settings()` and `trainable_training_config_from_trainer_settings()` provide discoverable conversion points for scripts and builders.

3. **Retrieval registry accepts object payloads but builders validate concrete payloads.** `RetrievalRegistry.build()` and `build_seed()` accept `object`; each builder calls `require_payload(payload, ExpectedPayload)` before reading fields. This keeps the registry interface uniform while making each builder's runtime dependency contract explicit.

4. **Checkpoint loader requires assembled dependencies.** The registry checkpoint builder creates default text/seed providers when callers do not inject them. `CheckpointGraphRetrieverLoader.load()` receives concrete providers and only loads model runtime state.

5. **Final retrieval returns a result object.** `RetrievalMethodResult` contains ranked nodes and a trace object. Flat methods use an empty trace; graph-aware methods populate retrieved edges. The seed ranker protocol remains separate because seed ranking requires full-node rankings, not top-k results.

## Risks / Trade-offs

- **Risk: broad signature churn in tests and methods** -> Mitigation: update retrieval contracts, flat/graph/trainable methods, execution service, and focused tests in one pass.
- **Risk: payload typing becomes too indirect** -> Mitigation: keep payload dataclasses small and method-specific, with clear `require_payload()` error messages.
- **Risk: legacy config helpers remain reachable** -> Mitigation: keep compatibility facade only where existing workflow code still imports it, and add architecture tests bounding usage.

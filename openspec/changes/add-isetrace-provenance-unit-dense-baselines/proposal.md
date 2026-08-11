## Why

The WSDM attribution study currently compares flat Dense/Dense-FT against provenance-path and provenance R-GCN methods that consume different candidate units. Without graph-free Dense variants over the exact provenance-unit candidate view, the reported gain cannot be separated into candidate-segmentation and graph contributions.

## What Changes

- Add `variant=flat|provenance_unit` to the existing public `dense` and `dense_ft` methods; retain `flat` as the default and restrict `provenance_unit` to ISETrace.
- Project ISETrace provenance content candidates into graph-free `TextRankingRequest` values for train, development, and test; neither variant receives a `ProvenanceGraph`, graph edge, graph identifier feature, or provenance retrieval trace.
- Map exact gold source spans to every overlapping provenance unit for Dense-FT positives, using the same natural-only trajectory split, negative-sampling recipe, task-local development selection, encoder, optimizer, epoch count, seed, and token-budget evaluation as flat Dense-FT.
- Persist the Dense-FT variant in pair/cache identity, run outputs, artifacts, and checkpoint metadata so flat and provenance-unit models cannot be silently exchanged, while retaining the existing SentenceTransformer model-directory format and trainer.
- Extend Hydra composition, closed method dispatch, workflow planning, inspection output, focused tests, architecture documentation, and ISETrace runbooks for the two controls.
- Preserve all existing flat Dense/Dense-FT, provenance path, provenance R-GCN, and evidence-dataset behavior.

## Capabilities

### New Capabilities

- `isetrace-provenance-unit-dense-retrieval`: Defines graph-free frozen and fine-tuned Dense variants over ISETrace provenance units, exact-span supervision, lifecycle isolation, and matched evaluation.

### Modified Capabilities

None. The repository currently has no promoted baseline specs under `openspec/specs/`.

## Impact

Affected surfaces include typed Dense/Dense-FT experiment configs; ISETrace training/view adapters; pair, model, retrieve, and workflow dispatch; Dense-FT checkpoint metadata; method inspection; artifact/cache provenance; focused adapter/config/workflow/retrieval tests; and maintained architecture/operations documentation. The change adds no public retrieval method ID, dependency, graph schema, encoder, loss, trainer backend, evaluation metric, or compatibility alias.

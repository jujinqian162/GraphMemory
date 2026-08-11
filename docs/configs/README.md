# Experiment configuration

Root: `configs/config.yaml`.

| Path | Ownership |
|---|---|
| `config.yaml` | composition, cache refresh, tracking, Hydra output layout |
| `dataset/*.yaml` | sources, capacities, validation |
| `profile/*.yaml` | split counts and trainable scale |
| `method/*.yaml` | one complete final-method contract |
| `stage/*.yaml` | shared stage fragments (e.g. Dense-FT training) |

Each job composes exactly one `method`. Dense and Dense-FT expose one candidate-view `method.variant` (`flat` by default; `provenance_unit` only on ISETrace). R-GCN configs expose one ablation `method.variant` (default `full_rgcn`). A multirun may sweep variants across independent jobs, but each resolved job contains one scalar variant; list-valued variants are rejected.

Non-training datasets may configure only a test split. Trainable methods fail fast unless train/dev/test are all available. `dataset/isetrace.yaml` names `trajectory_source`, `natural_query_source`, and explicit `trajectories.splits.<split>` trajectory counts. These values establish disjoint ownership, and every selected trajectory contributes its available authored natural queries. The `full` profile materializes that complete owned split; bounded profiles then cap total tasks per split, so `smoke` runs one task. The pinned trajectory revision is inferred from dataset registration. Review state remains outside the four-field natural-query record and must be frozen operationally before formal runs.

```powershell
uv run python experiment/inspect.py kind=configs
uv run python experiment/inspect.py kind=methods
uv run python experiment/inspect.py kind=variants
```

# Experiment configuration

Root: `configs/config.yaml`.

| Path | Ownership |
|---|---|
| `config.yaml` | composition, cache refresh, tracking, Hydra output layout |
| `dataset/*.yaml` | sources, capacities, validation |
| `profile/*.yaml` | split counts and trainable scale |
| `method/*.yaml` | one complete final-method contract |
| `stage/*.yaml` | shared stage fragments (e.g. Dense-FT training) |

Each job composes exactly one `method`. R-GCN configs expose one `method.variant` (default `full_rgcn`). List-valued variants are rejected.

Non-training datasets may configure only a test split. Trainable methods fail fast unless train/dev/test are all available. `dataset/isetrace.yaml` names `trajectory_source`, `natural_query_source`, and explicit `trajectories.splits.<split>.natural/template` counts. These values establish disjoint trajectory ownership: selected natural trajectories contribute available authored queries, while selected template trajectories contribute one template query each. The `full` profile materializes that complete owned split; bounded profiles then cap total tasks per split, so `smoke` runs one task. The pinned trajectory revision is inferred from dataset registration. Review state remains outside the four-field natural-query record and must be frozen operationally before formal runs.

```powershell
uv run python experiment/inspect.py kind=configs
uv run python experiment/inspect.py kind=methods
uv run python experiment/inspect.py kind=variants
```

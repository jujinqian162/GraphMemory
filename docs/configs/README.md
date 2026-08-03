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

Non-training datasets may configure only a test split. Trainable methods fail fast unless train/dev/test are all available. `dataset/isetrace.yaml` names `trajectory_source`, `natural_query_source`, and exact `queries.splits.<split>.natural/template` counts; the pinned trajectory revision and frozen natural ownership weights are inferred from dataset registration rather than configured by users. ISETrace ignores generic profile split caps, so smoke jobs override these explicit counts. Review state remains outside the four-field natural-query record and must be frozen operationally before formal runs.

```powershell
uv run python experiment/inspect.py kind=configs
uv run python experiment/inspect.py kind=methods
uv run python experiment/inspect.py kind=variants
```

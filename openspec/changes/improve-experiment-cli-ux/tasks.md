## 1. Workflow Contracts

- [x] 1.1 Add method-derived workflow selection in `graph_memory.experiment`, including flat, graph-rerank, and checkpoint-backed trainable methods.
- [x] 1.2 Add contiguous `--from` / `--to` stage range selection over the selected method workflow and retire explicit stage-list selection.
- [x] 1.3 Keep dependency validation fail-fast for graph-rerank tuned configs, train pairs, and trainable checkpoints when users start from downstream stages.

## 2. Config and Discovery

- [x] 2.1 Resolve experiment configs by name from `configs/experiments/` while still accepting explicit paths.
- [x] 2.2 Resolve trainable method training config names from `configs/training/<method>/`.
- [x] 2.3 Add resource-listing helpers for stages, methods, experiment configs, search-space configs, training configs, profiles, and recipes.
- [x] 2.4 Add `profile list` singular alias and include resolved split source, max examples, seed, and offset in profile discovery output.

## 3. CLI and Presentation

- [x] 3.1 Add `--method`, `--to`, and color controls to `init`, `plan`, and `run` where applicable.
- [x] 3.2 Add read-only discovery subcommands: `stages list`, `methods list`, `configs list`, `profiles list`, and `recipes list`.
- [x] 3.3 Render plan output as separated command blocks with script labels, one option per line, and optional ANSI color on option names.

## 4. Verification

- [x] 4.1 Add regression tests covering method-first workflows, stage ranges, config-name resolution, discovery output, and formatted plan output.
- [x] 4.2 Run targeted experiment-runner tests.
- [x] 4.3 Run repository validation required for this change.

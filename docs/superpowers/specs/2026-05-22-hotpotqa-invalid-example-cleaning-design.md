# HotpotQA Invalid Example Cleaning Design

Date: 2026-05-22

Status: Approved for implementation.

## Context

`scripts/prepare_dataset.py` owns raw dataset download, checksum verification, and local raw file placement. It should not mutate downloaded raw files or introduce dataset-specific semantic rules.

`scripts/prepare_hotpotqa.py` owns HotpotQA-specific interpretation and conversion from raw labeled examples into leakage-safe input and label artifacts. Invalid HotpotQA examples are therefore handled at this boundary, not in the raw download step.

## Behavior

`prepare_hotpotqa.py` will drop invalid raw examples by default before split sampling and artifact conversion.

An invalid example is any raw record that cannot be parsed as a HotpotQA example or cannot be converted into at least one valid memory task and label pair. Examples include malformed required fields, empty memory sentences, or supporting facts that cannot map to sentence nodes.

The script will provide an explicit strict-mode CLI switch to fail on the first invalid example instead of dropping it. Strict mode is for raw data auditing and debugging.

## Data Flow

```text
raw HotpotQA JSON list
  -> classify each raw example as valid or invalid
  -> keep valid examples in original relative order
  -> sample selected examples from valid examples
  -> convert selected examples into input and label artifacts
  -> validate output artifact contracts
  -> write artifacts and run summary
```

Cleaning happens before split sampling so requested split sizes are drawn from valid examples only. The script does not rewrite the raw input file and does not write a cleaned raw dataset artifact in Phase 1.

## Observability

The run summary will record:

- `raw_examples`
- `valid_examples`
- `invalid_examples_dropped`
- `selected_examples`
- `parsed_examples`
- `task_inputs`
- `task_labels`
- `invalid_example_reasons`, grouped by error message

Console logs should state how many invalid examples were dropped. If strict mode is enabled and an invalid example is found, the raised error should include the raw record index and the original parse or conversion failure.

## Error Handling

Default mode:

- Drops invalid examples.
- Continues if enough valid examples remain for the requested split.
- Fails if split sampling asks for more valid examples than available.

Strict mode:

- Fails on the first invalid raw example.
- Includes the raw record index in the error.

Artifact validation remains fail-fast. Validators still do not clean, repair, drop, sort, or infer artifact data.

## Testing

Add focused tests for:

- Default mode drops malformed and unconvertible HotpotQA examples before sampling.
- Run summary records valid, dropped, selected, and reason counts.
- Strict mode fails with the invalid raw record index.
- Existing valid fixture behavior remains unchanged.

# HotpotQA Invalid Example Cleaning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `scripts/prepare_hotpotqa.py` drop invalid raw HotpotQA examples by default while preserving a strict failure mode for raw data auditing.

**Architecture:** Keep `graph_memory.hotpotqa` fail-fast and unchanged. Add cleaning/classification at the `prepare_hotpotqa.py` CLI boundary before split sampling, then convert the selected valid examples with the existing parser and converter. Record dropped counts and grouped reasons in the run summary.

**Tech Stack:** Python 3.12, pytest, existing `graph_memory.io`, `graph_memory.hotpotqa`, `graph_memory.splits`, and observability helpers.

---

### Task 1: Default Drop Behavior

**Files:**
- Modify: `tests/test_phase1_real_cli_smoke.py`
- Modify: `scripts/prepare_hotpotqa.py`

- [ ] **Step 1: Write the failing test**

Add this test to `tests/test_phase1_real_cli_smoke.py`:

```python
def test_prepare_hotpotqa_drops_invalid_examples_before_sampling(tmp_path):
    raw_path = tmp_path / "raw.json"
    valid_first = {
        "_id": "valid-first",
        "question": "Where is the Eiffel Tower?",
        "answer": "Paris",
        "context": [["Eiffel Tower", ["The Eiffel Tower is in Paris."]]],
        "supporting_facts": [["Eiffel Tower", 0]],
    }
    malformed = {
        "question": "Missing id",
        "answer": "nowhere",
        "context": [["Missing", ["This record has no id."]]],
        "supporting_facts": [["Missing", 0]],
    }
    unconvertible = {
        "_id": "bad-support",
        "question": "Which support is missing?",
        "answer": "missing",
        "context": [["Known", ["Only this sentence exists."]]],
        "supporting_facts": [["Unknown", 0]],
    }
    valid_second = {
        "_id": "valid-second",
        "question": "Where is the Louvre?",
        "answer": "Paris",
        "context": [["Louvre", ["The Louvre is in Paris."]]],
        "supporting_facts": [["Louvre", 0]],
    }
    raw_path.write_text(json.dumps([valid_first, malformed, unconvertible, valid_second]), encoding="utf-8")

    task_inputs_path = tmp_path / "memory_tasks.input.json"
    labels_path = tmp_path / "memory_tasks.labels.json"

    assert prepare_hotpotqa.main(
        [
            "--input",
            str(raw_path),
            "--output_input",
            str(task_inputs_path),
            "--output_labels",
            str(labels_path),
            "--max_examples",
            "2",
            "--seed",
            "13",
            "--offset",
            "0",
        ]
    ) == 0

    task_inputs = read_json(task_inputs_path)
    summary = read_json(tmp_path / "memory_tasks.input.run_summary.json")

    assert {task_input["task_id"] for task_input in task_inputs} == {"hotpot_valid-first", "hotpot_valid-second"}
    assert summary["counts"]["raw_examples"] == 4
    assert summary["counts"]["valid_examples"] == 2
    assert summary["counts"]["invalid_examples_dropped"] == 2
    assert summary["counts"]["selected_examples"] == 2
    assert any("_id" in reason for reason in summary["counts"]["invalid_example_reasons"])
    assert any("supporting fact" in reason for reason in summary["counts"]["invalid_example_reasons"])
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_phase1_real_cli_smoke.py::test_prepare_hotpotqa_drops_invalid_examples_before_sampling -q -p no:cacheprovider --basetemp .pytest-tmp
```

Expected: FAIL because `prepare_hotpotqa.py` still fails on the malformed example.

- [ ] **Step 3: Write minimal implementation**

In `scripts/prepare_hotpotqa.py`, add a `strict` field to `PrepareHotpotQAArgs`, add `--strict_invalid_examples`, and add a helper that parses and converts each raw record to classify validity before sampling. Default mode returns only valid raw records and reason counts; strict mode raises `ValueError` with `index=<raw_index>`.

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_phase1_real_cli_smoke.py::test_prepare_hotpotqa_drops_invalid_examples_before_sampling -q -p no:cacheprovider --basetemp .pytest-tmp
```

Expected: PASS.

### Task 2: Strict Mode Failure

**Files:**
- Modify: `tests/test_phase1_real_cli_smoke.py`
- Modify: `scripts/prepare_hotpotqa.py`

- [ ] **Step 1: Write the failing test**

Add this test to `tests/test_phase1_real_cli_smoke.py`:

```python
def test_prepare_hotpotqa_strict_mode_fails_on_invalid_example(tmp_path):
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(
        json.dumps(
            [
                {
                    "_id": "valid",
                    "question": "Where is the Eiffel Tower?",
                    "answer": "Paris",
                    "context": [["Eiffel Tower", ["The Eiffel Tower is in Paris."]]],
                    "supporting_facts": [["Eiffel Tower", 0]],
                },
                {
                    "_id": "bad-support",
                    "question": "Which support is missing?",
                    "answer": "missing",
                    "context": [["Known", ["Only this sentence exists."]]],
                    "supporting_facts": [["Unknown", 0]],
                },
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="index=1"):
        prepare_hotpotqa.main(
            [
                "--input",
                str(raw_path),
                "--output_input",
                str(tmp_path / "memory_tasks.input.json"),
                "--output_labels",
                str(tmp_path / "memory_tasks.labels.json"),
                "--strict_invalid_examples",
            ]
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_phase1_real_cli_smoke.py::test_prepare_hotpotqa_strict_mode_fails_on_invalid_example -q -p no:cacheprovider --basetemp .pytest-tmp
```

Expected: FAIL because the CLI option does not exist yet or strict mode is not implemented.

- [ ] **Step 3: Write minimal implementation**

Complete the strict-mode branch in the same helper from Task 1. The raised error should include both `index=<raw_index>` and the original failure message.

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_phase1_real_cli_smoke.py::test_prepare_hotpotqa_strict_mode_fails_on_invalid_example -q -p no:cacheprovider --basetemp .pytest-tmp
```

Expected: PASS.

### Task 3: Regression Verification

**Files:**
- Modify: `docs/40-operations/commands.md`

- [ ] **Step 1: Update command documentation**

Add one sentence under "Prepare HotpotQA Splits" explaining that invalid raw examples are dropped by default and `--strict_invalid_examples` fails on the first invalid record.

- [ ] **Step 2: Run focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_phase1_real_cli_smoke.py tests/test_phase1_real_data_structures.py -q -p no:cacheprovider --basetemp .pytest-tmp
```

Expected: PASS.

- [ ] **Step 3: Run full tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider --basetemp .pytest-tmp
```

Expected: PASS.

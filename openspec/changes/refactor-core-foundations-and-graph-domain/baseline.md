## Batch 0 Baseline Evidence

Date: 2026-06-03

No production code was moved during this baseline capture. The only code added in Batch 0 is regression coverage under `tests/`.

## Commands

### Full Pytest Baseline

Attempted commands:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q --basetemp C:\tmp\graph-memory-refactor-baseline -p no:cacheprovider
.\.venv\Scripts\python.exe -m pytest tests -q --basetemp C:\tmp\graph-memory-refactor-baseline-20260603a -p no:cacheprovider
.\.venv\Scripts\python.exe -m pytest tests -q --basetemp .pytest-tmp\graph-memory-refactor-baseline -p no:cacheprovider
.\.venv\Scripts\python.exe -m pytest tests -q --basetemp report\tmp\pytest-baseline-refactor-20260603 -p no:cacheprovider
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

Observed baseline:

- `C:\tmp\...` basetemp attempts failed with `PermissionError: [WinError 5]` while creating the basetemp directory.
- Existing `.pytest-tmp/` is permission-locked on this machine.
- `report/tmp/...` is writable for ordinary file operations, but pytest creates temporary directories with `mkdir(mode=0o700)` on Windows; those directories become inaccessible and the session fails with `PermissionError: [WinError 5]`.
- Running without `--basetemp` failed when pytest tried to scan `C:\Users\jujin\AppData\Local\Temp\pytest-of-jujin`, also with `PermissionError: [WinError 5]`.
- Before the tmp-path setup failures dominate, pytest reports 81 passing tests and 65 setup errors, all tied to pytest temporary directory access.

### Batch 0 Focused Tests

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_core_refactor_baseline_contracts.py -q -p no:cacheprovider
```

Result:

```text
4 passed in 3.67s
```

### Type Check Baseline

Attempted commands:

```powershell
uv run basedpyright --outputjson --level error
$env:UV_CACHE_DIR='report/tmp/uv-cache-refactor'; uv run basedpyright --outputjson --level error
.\.venv\Scripts\basedpyright.exe --outputjson --level error
```

Observed baseline:

- `uv run basedpyright --outputjson --level error` failed before type checking because uv could not read `C:\Users\jujin\AppData\Local\uv\cache\sdists-v9\.git`.
- Using a workspace-local `UV_CACHE_DIR` avoided that cache path but uv then failed to query `.venv\Scripts\python.exe` with `PermissionError: [WinError 5]`.
- Direct `.venv\Scripts\basedpyright.exe --outputjson --level error` ran and reported 43 errors, dominated by missing imports for dependencies such as `rank_bm25`, `numpy`, `sentence_transformers`, `torch`, `pytest`, `PIL`, `tqdm`, and `typing_extensions` under that direct invocation.

### OpenSpec Strict Validation

Command:

```powershell
openspec validate --all --strict
```

Result:

```text
Totals: 9 passed, 0 failed (9 items)
```

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import cast

import pytest

from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.infrastructure.io import read_json, write_json
from graph_memory.retrieval.methods.memory_stream.artifact import importance_content_digest
from graph_memory.validation import ContractValidationError, validate_importance_artifact
from scripts.data import clean_importance


def _task(task_id: str = "hotpot_ms_1") -> MemoryTaskInput:
    scores = [2, 4, 4, 8]
    return {
        "task_id": task_id,
        "query": "query",
        "memory_items": [
            {
                "id": f"m{index}",
                "node_type": "document_sentence",
                "text": f"sentence {index}",
                "source": "source",
                "sentence_id": index,
                "position": index,
            }
            for index in range(len(scores))
        ],
    }


def _legacy_artifact(task: MemoryTaskInput, scores: list[int] | None = None) -> dict[str, object]:
    values = scores or [2, 4, 4, 8]
    return {
        "method": "memory_stream",
        "model": "gpt-5.4-mini",
        "prompt_version": "memory-stream-importance-v2",
        "generation": {"do_sample": False, "use_cache": True, "max_new_tokens": 2048},
        "tasks": [
            {
                "task_id": task["task_id"],
                "content_digest": importance_content_digest(task),
                "scores": {f"m{index}": score for index, score in enumerate(values)},
            }
        ],
    }


def test_rank_normalization_preserves_order_ties_and_is_idempotent() -> None:
    normalized = clean_importance.normalize_task_scores({"m0": 2, "m1": 4, "m2": 4, "m3": 8})

    assert normalized == {"m0": 1, "m1": 6, "m2": 6, "m3": 10}
    assert clean_importance.normalize_task_scores(normalized) == normalized


def test_rank_normalization_maps_constant_task_to_five() -> None:
    assert clean_importance.normalize_task_scores({"m0": 7, "m1": 7}) == {"m0": 5, "m1": 5}


def test_clean_legacy_artifact_produces_compact_schema_and_summary() -> None:
    task = _task()

    artifact, summary = clean_importance.clean_legacy_artifact(
        _legacy_artifact(task),
        [task],
        source_path=Path("legacy.json"),
        source_sha256="abc123",
    )

    assert artifact == {
        "schema_version": 1,
        "method": "memory_stream",
        "tasks": [
            {
                "task_id": task["task_id"],
                "content_digest": importance_content_digest(task),
                "scores": {"m0": 1, "m1": 6, "m2": 6, "m3": 10},
            }
        ],
    }
    source = cast(dict[str, object], summary["source"])
    legacy_metadata = cast(dict[str, object], source["legacy_metadata"])
    assert source["path"] == "legacy.json"
    assert source["sha256"] == "abc123"
    assert legacy_metadata["model"] == "gpt-5.4-mini"
    assert summary["counts"] == {"tasks": 1, "memory_items": 4, "constant_tasks": 0}
    validate_importance_artifact(artifact, [task])


def test_clean_legacy_artifact_rejects_task_and_node_mismatches() -> None:
    task = _task()
    legacy = _legacy_artifact(task)
    legacy_tasks = cast(list[dict[str, object]], legacy["tasks"])
    legacy_tasks[0]["task_id"] = "wrong"
    with pytest.raises(ContractValidationError, match="order"):
        _ = clean_importance.clean_legacy_artifact(
            legacy,
            [task],
            source_path=Path("legacy.json"),
            source_sha256="abc123",
        )

    legacy = _legacy_artifact(task)
    legacy_tasks = cast(list[dict[str, object]], legacy["tasks"])
    scores = cast(dict[str, int], legacy_tasks[0]["scores"])
    del scores["m3"]
    with pytest.raises(ContractValidationError, match="missing=.*m3"):
        _ = clean_importance.clean_legacy_artifact(
            legacy,
            [task],
            source_path=Path("legacy.json"),
            source_sha256="abc123",
        )


def test_strict_json_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    _ = path.write_text('{"method":"memory_stream","method":"other"}', encoding="utf-8")

    with pytest.raises(ContractValidationError, match="duplicate key=method"):
        _ = clean_importance.read_json_strict(path)


def test_cli_defaults_clean_first_1000_tasks_and_write_both_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    task = _task()
    tasks = [task, _task("unused")]
    write_json(Path("data/hotpotqa/processed/dev_memory_tasks.input.json"), tasks)
    write_json(
        Path("data/hotpotqa/processed/memory_stream/dev.first_1000.gpt-5.4-mini.importance.json"),
        _legacy_artifact(task),
    )

    assert clean_importance.main(["--limit", "1"]) == 0

    output = cast(
        dict[str, object],
        read_json(Path("data/hotpotqa/processed/memory_stream/dev.first_1000.importance.json")),
    )
    summary = cast(
        dict[str, object],
        read_json(
            Path("data/hotpotqa/processed/memory_stream/dev.first_1000.importance.cleaning_summary.json")
        ),
    )
    assert output["schema_version"] == 1
    assert set(output) == {"schema_version", "method", "tasks"}
    counts = cast(dict[str, object], summary["counts"])
    assert counts["tasks"] == 1


def test_cli_rejects_legacy_task_count_that_does_not_match_selected_prefix(
    tmp_path: Path,
) -> None:
    task = _task()
    tasks_path = tmp_path / "tasks.json"
    input_path = tmp_path / "legacy.json"
    write_json(tasks_path, [task, _task("second")])
    write_json(input_path, _legacy_artifact(task))

    with pytest.raises(ContractValidationError, match="task count mismatch"):
        _ = clean_importance.main(
            [
                "--tasks",
                str(tasks_path),
                "--input",
                str(input_path),
                "--limit",
                "2",
                "--output",
                str(tmp_path / "output.json"),
                "--summary",
                str(tmp_path / "summary.json"),
            ]
        )


def test_clean_importance_script_runs_directly_from_repository_root() -> None:
    repository_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [sys.executable, "scripts/data/clean_importance.py", "--help"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "task-rank normalize" in completed.stdout

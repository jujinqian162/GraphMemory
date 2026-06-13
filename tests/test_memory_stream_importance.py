from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Literal

import pytest

from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.infrastructure.io import write_json_atomic
from graph_memory.retrieval.methods.memory_stream.annotation import annotate_importance_tasks
from graph_memory.retrieval.methods.memory_stream.cache import ImportanceCache
from graph_memory.retrieval.methods.memory_stream.contracts import (
    GenerationResult,
    ImportanceArtifact,
    ImportanceMessage,
    ImportanceSettings,
    TaskImportanceRecord,
)
from graph_memory.retrieval.methods.memory_stream.prompt import (
    IMPORTANCE_PROMPT_VERSION,
    build_importance_messages,
    importance_cache_digest,
    importance_content_digest,
    parse_importance_response,
)
from graph_memory.retrieval.methods.memory_stream.runtime import LocalTransformersImportanceRuntime
from graph_memory.retrieval.methods.memory_stream.settings import ImportanceAnnotationSettings
from graph_memory.validation import (
    ContractValidationError,
    select_importance_records,
    validate_importance_artifact,
)


def _task(*, query: str = "QUERY_SENTINEL") -> MemoryTaskInput:
    return {
        "task_id": "hotpot_ms_1",
        "query": query,
        "memory_items": [
            {
                "id": "m0",
                "node_type": "document_sentence",
                "text": "The Eiffel Tower is in Paris.",
                "source": "Eiffel Tower",
                "sentence_id": 0,
                "position": 0,
            },
            {
                "id": "m1",
                "node_type": "document_sentence",
                "text": "The Seine runs through Paris.",
                "source": "Paris",
                "sentence_id": 0,
                "position": 1,
            },
        ],
        "metadata": {
            "gold_answer": "ANSWER_SENTINEL",
            "gold_evidence_nodes": ["GOLD_NODE_SENTINEL"],
        },
        "debug": {
            "graph": {
                "edges": [["GRAPH_SENTINEL", "m0"]],
            }
        },
    }


def _second_task() -> MemoryTaskInput:
    return {
        "task_id": "hotpot_ms_2",
        "query": "second query",
        "memory_items": [
            {
                "id": "m0",
                "node_type": "document_sentence",
                "text": "Mercury is the closest planet to the Sun.",
                "source": "Mercury",
                "sentence_id": 0,
                "position": 0,
            }
        ],
    }


def _settings(
    *,
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    model_path: Path | None = None,
    max_new_tokens: int = 256,
    device: Literal["auto", "cuda", "cpu"] = "auto",
) -> ImportanceAnnotationSettings:
    return ImportanceAnnotationSettings(
        model_id=model_id,
        model_path=model_path or Path("models/Qwen2.5-7B-Instruct"),
        prompt_version=IMPORTANCE_PROMPT_VERSION,
        device=device,
        trust_remote_code=True,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        tp_plan=None,
        do_sample=False,
        use_cache=True,
        max_new_tokens=max_new_tokens,
    )


class FakeRuntime:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.load_calls = 0
        self.generated_messages: list[list[ImportanceMessage]] = []

    def load(self) -> dict[str, object]:
        self.load_calls += 1
        return {"model_load_seconds": 0.25, "device": "fake"}

    def generate(
        self,
        messages: list[ImportanceMessage],
        settings: ImportanceSettings,
    ) -> GenerationResult:
        _ = settings
        self.generated_messages.append(messages)
        if not self.responses:
            raise AssertionError("unexpected generation call")
        return GenerationResult(
            text=self.responses.pop(0),
            generated_tokens=7,
            generation_seconds=0.5,
        )


def test_prompt_and_cache_key_exclude_query_labels_and_graph_values() -> None:
    task = _task()
    settings = _settings()

    messages = build_importance_messages(task, settings.prompt_version)
    digest = importance_cache_digest(task, settings)
    prompt_text = json.dumps(messages, ensure_ascii=False)
    semantic_payload = json.dumps(
        {
            "cache_digest": digest,
            "content_digest": importance_content_digest(task),
        },
        ensure_ascii=False,
    )

    for sentinel in ("QUERY_SENTINEL", "ANSWER_SENTINEL", "GOLD_NODE_SENTINEL", "GRAPH_SENTINEL"):
        assert sentinel not in prompt_text
        assert sentinel not in semantic_payload

    assert "The Eiffel Tower is in Paris." in prompt_text
    assert "m0" in prompt_text
    assert "position" in prompt_text


def test_cache_digest_tracks_semantic_inputs_but_not_runtime_placement() -> None:
    task = _task(query="first query")
    base = _settings(model_path=Path("models/original"), device="auto")
    moved = _settings(model_path=Path("models/moved"), device="cuda")
    changed_query = _task(query="different query")
    changed_model = _settings(model_id="Qwen/Qwen2.5-7B-Instruct-repacked")
    changed_generation = _settings(max_new_tokens=512)

    assert importance_cache_digest(task, base) == importance_cache_digest(task, moved)
    assert importance_cache_digest(task, base) == importance_cache_digest(changed_query, base)
    assert importance_cache_digest(task, base) != importance_cache_digest(task, changed_model)
    assert importance_cache_digest(task, base) != importance_cache_digest(task, changed_generation)

    reordered = _task()
    reordered["memory_items"] = list(reversed(reordered["memory_items"]))
    assert importance_content_digest(task) != importance_content_digest(reordered)


def test_response_parser_accepts_plain_and_fenced_json() -> None:
    task = _task()

    assert parse_importance_response('{"scores":[8,3]}', task) == {"m0": 8, "m1": 3}
    assert parse_importance_response('```json\n{"scores":[8,3]}\n```', task) == {"m0": 8, "m1": 3}


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ('{"scores":[8]}', "expected=2.*observed=1"),
        ('{"scores":[8,3,1]}', "expected=2.*observed=3"),
        ('{"scores":[true,3]}', "m0.*integer"),
        ('{"scores":[4.5,3]}', "m0.*integer"),
        ('{"scores":[0,3]}', "m0.*1-10"),
        ('{"scores":[11,3]}', "m0.*1-10"),
        ('{"scores":{"m0":8,"m1":3}}', "scores must be an array"),
        ('{"scores":[8,3],"fallback":4}', "unknown"),
    ],
)
def test_response_parser_rejects_non_exact_integer_coverage(payload: str, match: str) -> None:
    with pytest.raises(ContractValidationError, match=match):
        parse_importance_response(payload, _task())


def test_prompt_requests_scores_in_memory_item_order_without_node_id_copying() -> None:
    messages = build_importance_messages(_task(), IMPORTANCE_PROMPT_VERSION)
    system_prompt = messages[0]["content"]
    user_payload = json.loads(messages[1]["content"])

    assert user_payload["output_format"] == {
        "scores": ["<integer 1-10>", "<integer 1-10>", "..."]
    }
    assert "Do not return node ids." in system_prompt


def test_importance_artifact_validation_requires_task_order_node_coverage_and_digest() -> None:
    tasks = [_task(), _second_task()]
    artifact: ImportanceArtifact = {
        "method": "memory_stream",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "prompt_version": IMPORTANCE_PROMPT_VERSION,
        "generation": {"do_sample": False, "use_cache": True, "max_new_tokens": 256},
        "tasks": [
            {
                "task_id": "hotpot_ms_1",
                "content_digest": importance_content_digest(tasks[0]),
                "scores": {"m0": 7, "m1": 5},
            },
            {
                "task_id": "hotpot_ms_2",
                "content_digest": importance_content_digest(tasks[1]),
                "scores": {"m0": 9},
            },
        ],
    }

    validate_importance_artifact(artifact, tasks)

    out_of_order = {**artifact, "tasks": list(reversed(artifact["tasks"]))}
    with pytest.raises(ContractValidationError, match="order"):
        validate_importance_artifact(out_of_order, tasks)

    bad_digest = {
        **artifact,
        "tasks": [{**artifact["tasks"][0], "content_digest": "bad"}, artifact["tasks"][1]],
    }
    with pytest.raises(ContractValidationError, match="content_digest"):
        validate_importance_artifact(bad_digest, tasks)


def test_global_importance_artifact_selects_workflow_subset_by_task_id() -> None:
    tasks = [_task(), _second_task()]
    artifact: ImportanceArtifact = {
        "method": "memory_stream",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "prompt_version": IMPORTANCE_PROMPT_VERSION,
        "generation": {"do_sample": False, "use_cache": True, "max_new_tokens": 256},
        "tasks": [
            {
                "task_id": "hotpot_ms_1",
                "content_digest": importance_content_digest(tasks[0]),
                "scores": {"m0": 7, "m1": 5},
            },
            {
                "task_id": "hotpot_ms_2",
                "content_digest": importance_content_digest(tasks[1]),
                "scores": {"m0": 9},
            },
        ],
    }

    selected = select_importance_records(artifact, [tasks[1], tasks[0]])

    assert [record["task_id"] for record in selected] == ["hotpot_ms_2", "hotpot_ms_1"]


def test_global_importance_artifact_rejects_missing_duplicate_and_stale_subset_records() -> None:
    tasks = [_task(), _second_task()]
    first_record: TaskImportanceRecord = {
        "task_id": "hotpot_ms_1",
        "content_digest": importance_content_digest(tasks[0]),
        "scores": {"m0": 7, "m1": 5},
    }
    artifact: ImportanceArtifact = {
        "method": "memory_stream",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "prompt_version": IMPORTANCE_PROMPT_VERSION,
        "generation": {"do_sample": False, "use_cache": True, "max_new_tokens": 256},
        "tasks": [first_record],
    }

    with pytest.raises(ContractValidationError, match="missing task_id=hotpot_ms_2"):
        select_importance_records(artifact, [tasks[1]])

    duplicate = {**artifact, "tasks": [first_record, first_record]}
    with pytest.raises(ContractValidationError, match="duplicate task_id=hotpot_ms_1"):
        select_importance_records(duplicate, [tasks[0]])

    stale_task = _task()
    stale_task["memory_items"][0] = {
        **stale_task["memory_items"][0],
        "text": "Changed content.",
    }
    with pytest.raises(ContractValidationError, match="content_digest"):
        select_importance_records(artifact, [stale_task])


def test_duplicate_memory_node_ids_are_rejected_before_prompting() -> None:
    task = _task()
    task["memory_items"][1] = {**task["memory_items"][1], "id": "m0"}

    with pytest.raises(ContractValidationError, match="duplicate.*m0"):
        build_importance_messages(task, IMPORTANCE_PROMPT_VERSION)


def test_write_json_atomic_replaces_existing_json_without_temp_files(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"

    write_json_atomic(target, {"version": 1})
    write_json_atomic(target, {"version": 2})

    assert json.loads(target.read_text(encoding="utf-8")) == {"version": 2}
    assert list(tmp_path.glob("*.tmp")) == []


def test_importance_cache_reuses_valid_records_and_treats_corruption_as_miss(tmp_path: Path) -> None:
    task = _task()
    settings = _settings()
    cache = ImportanceCache(tmp_path)
    record: TaskImportanceRecord = {
        "task_id": task["task_id"],
        "content_digest": importance_content_digest(task),
        "scores": {"m0": 8, "m1": 4},
    }

    path = cache.write(task, settings, record)
    digest = importance_cache_digest(task, settings)
    assert path == tmp_path / digest[:2] / f"{digest}.json"
    assert cache.read(task, settings) == record
    assert cache.read(task, _settings(max_new_tokens=512)) is None

    cache.path_for_task(task, settings).write_text("{not-json", encoding="utf-8")
    assert cache.read(task, settings) is None


def test_annotation_uses_no_runtime_when_every_task_is_cached(tmp_path: Path) -> None:
    task = _task()
    settings = _settings()
    cache = ImportanceCache(tmp_path)
    cache.write(
        task,
        settings,
        {
            "task_id": task["task_id"],
            "content_digest": importance_content_digest(task),
            "scores": {"m0": 8, "m1": 4},
        },
    )

    def fail_factory(_settings: ImportanceSettings) -> FakeRuntime:
        raise AssertionError("runtime must not be created on all-cache-hit runs")

    result = annotate_importance_tasks([task], settings, cache_dir=tmp_path, runtime_factory=fail_factory)

    assert result.cache_stats.hits == 1
    assert result.model_load_count == 0
    assert result.generation_calls == 0
    assert result.artifact["tasks"][0]["scores"] == {"m0": 8, "m1": 4}


def test_annotation_loads_one_runtime_for_all_misses_and_preserves_input_order(tmp_path: Path) -> None:
    tasks = [_task(), _second_task()]
    runtime = FakeRuntime(
        [
            '{"scores":[8,4]}',
            '{"scores":[6]}',
        ]
    )
    created: list[FakeRuntime] = []

    def factory(_settings: ImportanceSettings) -> FakeRuntime:
        created.append(runtime)
        return runtime

    result = annotate_importance_tasks(tasks, _settings(), cache_dir=tmp_path, runtime_factory=factory)

    assert created == [runtime]
    assert runtime.load_calls == 1
    assert len(runtime.generated_messages) == 2
    assert result.model_load_count == 1
    assert result.generation_calls == 2
    assert result.generated_tokens == 14
    assert result.generation_seconds == pytest.approx(1.0)
    assert [record["task_id"] for record in result.artifact["tasks"]] == ["hotpot_ms_1", "hotpot_ms_2"]
    validate_importance_artifact(result.artifact, tasks)


def test_annotation_keeps_successful_cache_after_later_invalid_output(tmp_path: Path) -> None:
    tasks = [_task(), _second_task()]
    runtime = FakeRuntime(
        [
            '{"scores":[8,4]}',
            '{"scores":[true]}',
        ]
    )
    created: list[FakeRuntime] = []

    with pytest.raises(ContractValidationError, match="hotpot_ms_2.*m0.*integer"):
        annotate_importance_tasks(
            tasks,
            _settings(),
            cache_dir=tmp_path,
            runtime_factory=lambda _settings: created.append(runtime) or runtime,
        )

    assert created == [runtime]
    assert runtime.load_calls == 1
    assert len(runtime.generated_messages) == 2
    cache = ImportanceCache(tmp_path)
    assert cache.read(tasks[0], _settings()) is not None
    assert cache.read(tasks[1], _settings()) is None


def test_local_runtime_uses_lazy_direct_transformers_path_without_server_patterns() -> None:
    source = inspect.getsource(LocalTransformersImportanceRuntime)

    assert "AutoTokenizer.from_pretrained" in source
    assert "AutoModelForCausalLM.from_pretrained" in source
    assert "tp_plan=None" in source
    assert "apply_chat_template" in source
    assert "torch.inference_mode" in source
    for forbidden in ("requests", "openai", "vllm", "ThreadPoolExecutor", "tensor_parallel"):
        assert forbidden not in source

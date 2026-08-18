from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, TypeVar, cast

from pydantic import BaseModel

from graph_memory.datasets.isetrace.end_to_end_qa.contracts import (
    ANSWER_SCHEMA,
    JUDGE_SCHEMA,
    AnswerArtifact,
    AnswerResponse,
    Condition,
    JudgeResponse,
    JudgmentArtifact,
    PreparedRecord,
)
from graph_memory.datasets.isetrace.end_to_end_qa.preparation import (
    CONDITIONS,
    EVIDENCE_TOKEN_BUDGET,
    iter_selected_tasks,
    load_frozen_inputs,
    load_gold_spans,
    load_rankings,
    prepare_task_conditions,
    sha256_file,
)
from graph_memory.datasets.isetrace.end_to_end_qa.prompts import (
    ANSWER_PROMPT_VERSION,
    JUDGE_PROMPT_VERSION,
    JUDGE_SYSTEM_PROMPT,
    answer_payload,
    answer_system_prompt,
    judge_payload,
    validate_answer,
    validate_judgment,
)
from graph_memory.datasets.isetrace.end_to_end_qa.reporting import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    build_report,
    write_summary_csv,
)
from graph_memory.infrastructure.io import write_json_atomic
from graph_memory.experiment.config import ResolvedExperimentConfig
from graph_memory.experiment.persistence import read_yaml_model
from graph_memory.infrastructure.responses import (
    ResponsesSettings,
    complete_structured_request,
    load_responses_settings,
    structured_request_body,
)
from graph_memory.text.chunking import load_offset_tokenizer

SEED = 13
_JsonlModelT = TypeVar("_JsonlModelT", bound=BaseModel)
_ArtifactT = TypeVar("_ArtifactT", AnswerArtifact, JudgmentArtifact)
_InputT = TypeVar("_InputT", bound=BaseModel)
_OutputT = TypeVar("_OutputT", bound="RecordArtifact")


class RecordArtifact(Protocol):
    @property
    def record_id(self) -> str: ...


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def dataset(self) -> Path:
        return (
            self.root
            / "data/processed/datasets/isetrace/a729232bf650444a89572949522d8941"
        )

    @property
    def task_ids(self) -> Path:
        return (
            self.root
            / "results/isetrace/e4-human-validation/human_quality_pass_task_ids.json"
        )

    @property
    def metadata(self) -> Path:
        return (
            self.root
            / "results/isetrace/e4-human-validation/human_corrected_query_metadata.jsonl"
        )

    @property
    def review_packet(self) -> Path:
        return self.root / "data/isetrace/e4-human-validation/review_packet.jsonl"

    @property
    def tokenizer(self) -> Path:
        return self.root / "models/intfloat-e5-base-v2"

    @property
    def output(self) -> Path:
        return self.root / "results/isetrace/e10-end-to-end-qa"

    @property
    def rankings(self) -> dict[Condition, Path]:
        return {
            Condition.FLAT_DENSE_FT: self.root
            / "results/isetrace/evalv8/runs/dense-ft/flat/isetrace_v7_dense_ft_flat_s13_evalv8/predictions/ranked_prefix.jsonl.gz",
            Condition.PU_DENSE_FT: self.root
            / "results/isetrace/evalv8/runs/dense-ft/provenance-unit/isetrace_v7_dense_ft_pu_s13_evalv8/predictions/ranked_prefix.jsonl.gz",
            Condition.RESIDUAL_RGCN: self.root
            / "results/isetrace/evalv8/runs/rgcn/residual/isetrace_v7_pu_dense_ft_rgcn_full_s13_evalv8/predictions/ranked_prefix.jsonl.gz",
        }

    @property
    def ranking_configs(self) -> dict[Condition, Path]:
        return {
            condition: path.parents[1] / "config/resolved.yaml"
            for condition, path in self.rankings.items()
        }


@dataclass(frozen=True)
class Options:
    action: Literal["prepare", "answer", "judge", "report"]
    workers: int
    limit: int | None


def main(root: Path) -> int:
    paths = Paths(root.resolve())
    options = _parse_args()
    if options.action == "prepare":
        prepare(paths, limit=options.limit)
    elif options.action == "answer":
        answer(paths, workers=options.workers, limit=options.limit)
    elif options.action == "judge":
        judge(paths, workers=options.workers, limit=options.limit)
    else:
        report(paths)
    return 0


def prepare(paths: Paths, *, limit: int | None) -> None:
    task_ids, metadata, packets = load_frozen_inputs(
        task_ids_path=paths.task_ids,
        metadata_path=paths.metadata,
        review_packet_path=paths.review_packet,
    )
    selected = set(task_ids[:limit] if limit is not None else task_ids)
    gold_spans = load_gold_spans(paths.dataset / "labels.json", task_ids=selected)
    ranking_methods = {
        Condition.FLAT_DENSE_FT: "dense_ft",
        Condition.PU_DENSE_FT: "dense_ft",
        Condition.RESIDUAL_RGCN: "provenance_rgcn",
    }
    expected_variants = {
        Condition.FLAT_DENSE_FT: "flat",
        Condition.PU_DENSE_FT: "provenance_unit",
        Condition.RESIDUAL_RGCN: "full_rgcn",
    }
    for condition, config_path in paths.ranking_configs.items():
        config = read_yaml_model(config_path, ResolvedExperimentConfig)
        if (
            config.seed != SEED
            or config.dataset.name != "isetrace"
            or config.method_id.value != ranking_methods[condition]
            or config.variant != expected_variants[condition]
        ):
            message = (
                "resolved retrieval config does not match frozen E10 "
                f"condition={condition.value}"
            )
            raise ValueError(message)
    rankings = {
        condition: load_rankings(
            path,
            task_ids=selected,
            expected_method=ranking_methods[condition],
        )
        for condition, path in paths.rankings.items()
    }
    tokenizer = load_offset_tokenizer(paths.tokenizer)
    records: list[PreparedRecord] = []
    for task in iter_selected_tasks(paths.dataset / "tasks.json", task_ids=selected):
        records.extend(
            prepare_task_conditions(
                task,
                metadata=metadata[task.task_id],
                review_packet=packets[task.task_id],
                gold_spans=gold_spans[task.task_id],
                rankings={
                    condition: condition_rankings[task.task_id]
                    for condition, condition_rankings in rankings.items()
                },
                tokenizer=tokenizer,
            )
        )
    records.sort(key=lambda record: record.record_id)
    prepared_path = paths.output / "prepared.jsonl"
    _write_jsonl_atomic(prepared_path, records)
    inputs = (
        paths.task_ids,
        paths.metadata,
        paths.review_packet,
        paths.dataset / "tasks.json",
        paths.dataset / "labels.json",
        *paths.rankings.values(),
        *paths.ranking_configs.values(),
    )
    write_json_atomic(
        paths.output / "manifest.json",
        {
            "schema_version": 1,
            "experiment": "ISETrace E10 End-to-End QA",
            "seed": SEED,
            "task_count": len(selected),
            "record_count": len(records),
            "conditions": [condition.value for condition in CONDITIONS],
            "evidence_token_budget": EVIDENCE_TOKEN_BUDGET,
            "tokenizer_path": paths.tokenizer.relative_to(paths.root).as_posix(),
            "inputs": {
                path.relative_to(paths.root).as_posix(): sha256_file(path)
                for path in inputs
            },
            "prepared_sha256": sha256_file(prepared_path),
            "bootstrap": {
                "unit": "trajectory_id",
                "samples": BOOTSTRAP_SAMPLES,
                "seed": BOOTSTRAP_SEED,
            },
        },
    )
    print(f"prepared {len(selected)} tasks / {len(records)} records at {prepared_path}")


def answer(paths: Paths, *, workers: int, limit: int | None) -> None:
    settings = load_responses_settings(paths.root / ".env")
    prepared = _load_prepared(paths)
    existing = _read_jsonl(paths.output / "answers.jsonl", AnswerArtifact)
    current = _current_artifacts(
        prepared,
        existing,
        input_digest=lambda record: _stage_input_digest(
            record,
            stage="answer",
            prompt_version=ANSWER_PROMPT_VERSION,
            settings=settings,
        ),
    )
    eligible = _limit_tasks(prepared, limit)
    pending = [record for record in eligible if record.record_id not in current]

    def request(record: PreparedRecord) -> AnswerArtifact:
        body = structured_request_body(
            settings=settings,
            system_prompt=answer_system_prompt(record),
            user_payload=answer_payload(record),
            schema_name="isetrace_e10_answer",
            schema=ANSWER_SCHEMA,
            max_output_tokens=1024,
            prompt_cache_key=f"gm:{ANSWER_PROMPT_VERSION}:{settings.model_id}",
        )
        response = complete_structured_request(
            body,
            settings=settings,
            cache_dir=paths.output / "cache/answer",
            validate=validate_answer,
        )
        return AnswerArtifact(
            record_id=record.record_id,
            input_digest=_stage_input_digest(
                record,
                stage="answer",
                prompt_version=ANSWER_PROMPT_VERSION,
                settings=settings,
            ),
            answer=AnswerResponse.model_validate(response.value),
            request_digest=response.request_digest,
            response_id=response.response_id,
            usage=response.usage,
            cached=response.cached,
        )

    _run_parallel(
        pending,
        current=current,
        workers=workers,
        output_path=paths.output / "answers.jsonl",
        request=request,
    )
    _update_model_manifest(paths, settings, stage="answer")


def judge(paths: Paths, *, workers: int, limit: int | None) -> None:
    settings = load_responses_settings(paths.root / ".env")
    prepared = _load_prepared(paths)
    answers = _read_jsonl(paths.output / "answers.jsonl", AnswerArtifact)
    answers_by_id = _current_artifacts(
        prepared,
        answers,
        input_digest=lambda record: _stage_input_digest(
            record,
            stage="answer",
            prompt_version=ANSWER_PROMPT_VERSION,
            settings=settings,
        ),
    )
    eligible = _limit_tasks(prepared, limit)
    missing_answers = [
        record.record_id for record in eligible if record.record_id not in answers_by_id
    ]
    if missing_answers:
        message = (
            "judge requires current answers for every selected condition; "
            f"missing {len(missing_answers)} selected answers"
        )
        raise ValueError(message)
    existing = _read_jsonl(paths.output / "judgments.jsonl", JudgmentArtifact)

    def digest(record: PreparedRecord) -> str:
        return _stage_input_digest(
            {
                "prepared": record,
                "answer": answers_by_id[record.record_id],
            },
            stage="judge",
            prompt_version=JUDGE_PROMPT_VERSION,
            settings=settings,
        )

    prepared_with_answers = [
        record for record in prepared if record.record_id in answers_by_id
    ]
    current = _current_artifacts(
        prepared_with_answers,
        existing,
        input_digest=digest,
    )
    pending = [record for record in eligible if record.record_id not in current]

    def request(record: PreparedRecord) -> JudgmentArtifact:
        answer_artifact = answers_by_id[record.record_id]
        body = structured_request_body(
            settings=settings,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_payload=judge_payload(record, answer_artifact.answer),
            schema_name="isetrace_e10_judgment",
            schema=JUDGE_SCHEMA,
            max_output_tokens=1024,
            prompt_cache_key=f"gm:{JUDGE_PROMPT_VERSION}:{settings.model_id}",
        )
        response = complete_structured_request(
            body,
            settings=settings,
            cache_dir=paths.output / "cache/judge",
            validate=validate_judgment,
        )
        return JudgmentArtifact(
            record_id=record.record_id,
            input_digest=digest(record),
            judgment=JudgeResponse.model_validate(response.value),
            request_digest=response.request_digest,
            response_id=response.response_id,
            usage=response.usage,
            cached=response.cached,
        )

    _run_parallel(
        pending,
        current=current,
        workers=workers,
        output_path=paths.output / "judgments.jsonl",
        request=request,
    )
    _update_model_manifest(paths, settings, stage="judge")
    judgments_path = paths.output / "judgments.jsonl"
    if len(current) == len(prepared):
        _update_judgment_manifest(paths, judgments_path)


def report(paths: Paths) -> None:
    prepared = _load_prepared(paths)
    judgments_path = paths.output / "judgments.jsonl"
    manifest = _read_json_object(paths.output / "manifest.json")
    judgments_identity_value = manifest.get("judgments")
    if not isinstance(judgments_identity_value, dict):
        raise ValueError("manifest has no completed judgment identity")
    judgments_identity = _string_object_mapping(
        cast(Mapping[object, object], judgments_identity_value)
    )
    if judgments_identity.get("prepared_sha256") != sha256_file(
        paths.output / "prepared.jsonl"
    ):
        raise ValueError("judgments were not produced for the current prepared inputs")
    if judgments_identity.get("judgments_sha256") != sha256_file(judgments_path):
        raise ValueError("judgment artifact digest disagrees with manifest")
    judgments = _read_jsonl(judgments_path, JudgmentArtifact)
    result = build_report(prepared, judgments)
    write_json_atomic(paths.output / "report.json", result)
    write_summary_csv(paths.output / "summary.csv", result)
    print(f"wrote {paths.output / 'report.json'} and {paths.output / 'summary.csv'}")


def _parse_args() -> Options:
    parser = argparse.ArgumentParser(description="Run frozen ISETrace E10 QA")
    _ = parser.add_argument(
        "--action",
        choices=("prepare", "answer", "judge", "report"),
        required=True,
    )
    _ = parser.add_argument("--workers", type=int, default=8)
    _ = parser.add_argument("--limit", type=int)
    namespace_values = vars(parser.parse_args())
    action = namespace_values.get("action")
    workers = namespace_values.get("workers")
    limit = namespace_values.get("limit")
    if not isinstance(action, str) or action not in {
        "prepare",
        "answer",
        "judge",
        "report",
    }:
        parser.error("invalid --action")
    if not isinstance(workers, int) or workers <= 0:
        parser.error("--workers must be positive")
    if limit is not None and (not isinstance(limit, int) or limit <= 0):
        parser.error("--limit must be positive")
    if action == "report" and limit is not None:
        parser.error("--limit is not valid for report")
    return Options(
        action=cast(Literal["prepare", "answer", "judge", "report"], action),
        workers=workers,
        limit=limit,
    )


def _limit_tasks(
    records: Sequence[PreparedRecord],
    limit: int | None,
) -> list[PreparedRecord]:
    if limit is None:
        return list(records)
    task_ids: set[str] = set()
    for record in records:
        task_ids.add(record.task_id)
        if len(task_ids) == limit:
            break
    return [record for record in records if record.task_id in task_ids]


def _run_parallel(
    records: Sequence[_InputT],
    *,
    current: dict[str, _OutputT],
    workers: int,
    output_path: Path,
    request: Callable[[_InputT], _OutputT],
) -> None:
    if not records:
        print(f"nothing pending for {output_path}")
        return
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = {executor.submit(request, record): record for record in records}
    first_error: BaseException | None = None
    for future in as_completed(futures):
        try:
            result = future.result()
        except BaseException as error:
            if first_error is None:
                first_error = error
        else:
            current[result.record_id] = result
            print(f"{output_path.name}: {len(current)} complete", file=sys.stderr)
    executor.shutdown(wait=True)
    _write_jsonl_atomic(output_path, current.values())
    if first_error is not None:
        raise first_error


def _load_prepared(paths: Paths) -> list[PreparedRecord]:
    prepared_path = paths.output / "prepared.jsonl"
    records = _read_jsonl(prepared_path, PreparedRecord)
    manifest_value = _read_json_object(paths.output / "manifest.json")
    if manifest_value.get("record_count") != len(records):
        raise ValueError("prepared record count disagrees with manifest")
    if manifest_value.get("prepared_sha256") != sha256_file(prepared_path):
        raise ValueError("prepared input digest disagrees with manifest")
    return records


def _current_artifacts(
    prepared: Sequence[PreparedRecord],
    existing: Sequence[_ArtifactT],
    *,
    input_digest: Callable[[PreparedRecord], str],
) -> dict[str, _ArtifactT]:
    prepared_by_id = {record.record_id: record for record in prepared}
    current: dict[str, _ArtifactT] = {}
    for artifact in existing:
        record_id = artifact.record_id
        record = prepared_by_id.get(record_id)
        if record is not None and artifact.input_digest == input_digest(record):
            current[record_id] = artifact
    return current


def _stage_input_digest(
    value: object,
    *,
    stage: str,
    prompt_version: str,
    settings: ResponsesSettings,
) -> str:
    record = {
        "stage": stage,
        "prompt_version": prompt_version,
        "model_id": settings.model_id,
        "base_url": settings.base_url.rstrip("/"),
        "input": _json_value(value),
    }
    encoded = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _update_model_manifest(
    paths: Paths,
    settings: ResponsesSettings,
    *,
    stage: Literal["answer", "judge"],
) -> None:
    manifest_path = paths.output / "manifest.json"
    manifest_value = _read_json_object(manifest_path)
    models_value = manifest_value.setdefault("models", {})
    if not isinstance(models_value, dict):
        raise ValueError("manifest models must be an object")
    models_value[stage] = {
        "model_id": settings.model_id,
        "endpoint_sha256": hashlib.sha256(
            settings.base_url.rstrip("/").encode("utf-8")
        ).hexdigest(),
        "prompt_version": (
            ANSWER_PROMPT_VERSION if stage == "answer" else JUDGE_PROMPT_VERSION
        ),
    }
    write_json_atomic(manifest_path, manifest_value)


def _update_judgment_manifest(paths: Paths, judgments_path: Path) -> None:
    manifest_path = paths.output / "manifest.json"
    manifest = _read_json_object(manifest_path)
    manifest["judgments"] = {
        "prepared_sha256": sha256_file(paths.output / "prepared.jsonl"),
        "judgments_sha256": sha256_file(judgments_path),
    }
    write_json_atomic(manifest_path, manifest)


def _string_object_mapping(value: Mapping[object, object]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError("JSON object keys must be strings")
        output[key] = item
    return output


def _read_json_object(path: Path) -> dict[str, object]:
    value = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    mapping = cast(Mapping[object, object], value)
    output: dict[str, object] = {}
    for key, item in mapping.items():
        if not isinstance(key, str):
            raise ValueError(f"{path} must contain only string object keys")
        output[key] = item
    return output


def _read_jsonl(path: Path, model: type[_JsonlModelT]) -> list[_JsonlModelT]:
    if not path.exists():
        return []
    with path.open("rb") as stream:
        return [model.model_validate_json(line) for line in stream]


def _write_jsonl_atomic(
    path: Path,
    records: Iterable[RecordArtifact],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    ordered = sorted(records, key=lambda record: record.record_id)
    with temporary.open("wb") as stream:
        for record in ordered:
            if not isinstance(record, BaseModel):
                raise TypeError("JSONL artifact must be a Pydantic model")
            _ = stream.write(record.model_dump_json().encode("utf-8"))
            _ = stream.write(b"\n")
    _ = temporary.replace(path)


def _json_value(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        mapping = cast(Mapping[object, object], value)
        output: dict[str, object] = {}
        for key, item in mapping.items():
            output[str(key)] = _json_value(item)
        return output
    if isinstance(value, (list, tuple)):
        sequence = cast(Sequence[object], value)
        return [_json_value(item) for item in sequence]
    return value


__all__ = ["Paths", "answer", "judge", "main", "prepare", "report"]

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from http.client import HTTPResponse
from pathlib import Path
from typing import cast

from graph_memory.infrastructure.io import write_json_atomic


@dataclass(frozen=True)
class ResponsesSettings:
    model_id: str
    api_key: str
    base_url: str


@dataclass(frozen=True)
class StructuredResponse:
    value: dict[str, object]
    request_digest: str
    response_id: str | None
    usage: dict[str, object]
    cached: bool


@dataclass(frozen=True)
class _CacheRecord:
    value: dict[str, object]
    response_id: str | None
    usage: dict[str, object]


_RATE_LIMIT_LOCK = threading.Lock()
_RATE_LIMIT_INTERVALS: dict[tuple[str, str], float] = {}
_RATE_LIMIT_NEXT_REQUESTS: dict[tuple[str, str], float] = {}


def load_responses_settings(path: Path) -> ResponsesSettings:
    values: dict[str, str] = {}
    if path.exists():
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            values[key.strip()] = value
    for key in ("MODEL_ID", "API_KEY", "BASE_URL"):
        environment_value = os.environ.get(key)
        if environment_value:
            values[key] = environment_value
        if not values.get(key):
            raise ValueError(f"missing {key} in environment or {path}")
    return ResponsesSettings(
        model_id=values["MODEL_ID"],
        api_key=values["API_KEY"],
        base_url=values["BASE_URL"],
    )


def structured_request_body(
    *,
    settings: ResponsesSettings,
    system_prompt: str,
    user_payload: object,
    schema_name: str,
    schema: dict[str, object],
    max_output_tokens: int,
    prompt_cache_key: str,
) -> dict[str, object]:
    return {
        "model": settings.model_id,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": json.dumps(
                            user_payload,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    }
                ],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
        "max_output_tokens": max_output_tokens,
        "store": False,
        "stream": False,
        "prompt_cache_key": prompt_cache_key,
    }


def complete_structured_request(
    body: dict[str, object],
    *,
    settings: ResponsesSettings,
    cache_dir: Path,
    validate: Callable[[object], dict[str, object]],
    timeout_seconds: float = 180.0,
    transport_retries: int = 5,
    validation_retries: int = 2,
) -> StructuredResponse:
    request_digest = _digest({"base_url": settings.base_url.rstrip("/"), "body": body})
    cache_path = cache_dir / f"{request_digest}.json"
    cached = _read_cache(cache_path, validate=validate)
    if cached is not None:
        return StructuredResponse(
            value=cached.value,
            request_digest=request_digest,
            response_id=cached.response_id,
            usage=cached.usage,
            cached=True,
        )

    last_error: Exception | None = None
    request_body = body
    for validation_attempt in range(validation_retries + 1):
        response = _post_response(
            request_body,
            settings=settings,
            timeout_seconds=timeout_seconds,
            max_retries=transport_retries,
        )
        try:
            value = validate(_response_json(response))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            last_error = ValueError(f"{error}; {_response_diagnostic(response)}")
            if validation_attempt < validation_retries:
                request_body = _validation_retry_body(body)
                time.sleep(min(2.0**validation_attempt, 30.0))
            continue
        response_id_value = response.get("id")
        response_id = response_id_value if isinstance(response_id_value, str) else None
        usage = _string_mapping(response.get("usage")) or {}
        write_json_atomic(
            cache_path,
            {
                "request_digest": request_digest,
                "response_id": response_id,
                "usage": usage,
                "value": value,
            },
        )
        return StructuredResponse(
            value=value,
            request_digest=request_digest,
            response_id=response_id,
            usage=usage,
            cached=False,
        )
    message = (
        "Responses API returned invalid structured output after "
        f"{validation_retries + 1} attempts: {last_error}"
    )
    raise RuntimeError(message) from last_error


def _read_cache(
    path: Path,
    *,
    validate: Callable[[object], dict[str, object]],
) -> _CacheRecord | None:
    if not path.exists():
        return None
    try:
        record_value = cast(object, json.loads(path.read_text(encoding="utf-8")))
        record = _json_object(record_value)
        value = validate(record.get("value"))
        response_id_value = record.get("response_id")
        response_id = response_id_value if isinstance(response_id_value, str) else None
        usage = _string_mapping(record.get("usage")) or {}
        return _CacheRecord(value=value, response_id=response_id, usage=usage)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        path.unlink(missing_ok=True)
        return None


def _post_response(
    body: dict[str, object],
    *,
    settings: ResponsesSettings,
    timeout_seconds: float,
    max_retries: int,
) -> dict[str, object]:
    endpoint = settings.base_url.rstrip("/")
    if not endpoint.endswith("/responses"):
        endpoint += "/responses"
    payload = json.dumps(body).encode("utf-8")
    transient_attempt = 0
    rate_limit_key = (endpoint, settings.model_id)
    while True:
        _wait_for_rate_limit_slot(rate_limit_key)
        request = urllib.request.Request(
            endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {settings.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "GraphMemory/1.0",
            },
            method="POST",
        )
        try:
            response = cast(
                HTTPResponse,
                urllib.request.urlopen(request, timeout=timeout_seconds),
            )
            with response:
                value = cast(object, json.loads(response.read()))
                return _json_object(value)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:1000]
            if error.code == 429:
                retry_after = error.headers.get("Retry-After")
                delay, requests_per_minute = _register_rate_limit(
                    rate_limit_key,
                    detail=detail,
                    retry_after=retry_after,
                )
                rate_limit_message = (
                    "Responses API rate limited; "
                    f"waiting at least {delay:.1f}s and pacing this model at "
                    f"{requests_per_minute} RPM"
                )
                print(
                    rate_limit_message,
                    file=sys.stderr,
                    flush=True,
                )
                continue
            retryable_gateway_403 = error.code == 403 and (
                "bad_response_status_code" in detail
                or "please try again" in detail.lower()
            )
            retryable = error.code >= 500 or retryable_gateway_403
            if not retryable or transient_attempt >= max_retries:
                raise RuntimeError(
                    f"Responses API HTTP {error.code}: {detail}"
                ) from error
            retry_after_value = error.headers.get("Retry-After")
            retry_after = (
                retry_after_value if isinstance(retry_after_value, str) else None
            )
            try:
                delay = float(retry_after) if retry_after else 2.0**transient_attempt
            except ValueError:
                delay = 2.0**transient_attempt
            transient_attempt += 1
            time.sleep(min(delay, 30.0))
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            ValueError,
        ) as error:
            if transient_attempt >= max_retries:
                raise RuntimeError(
                    f"Responses API failed after {max_retries + 1} attempts: {error}"
                ) from error
            time.sleep(min(2.0**transient_attempt, 30.0))
            transient_attempt += 1


def _wait_for_rate_limit_slot(key: tuple[str, str]) -> None:
    with _RATE_LIMIT_LOCK:
        now = time.monotonic()
        interval = _RATE_LIMIT_INTERVALS.get(key, 0.0)
        request_at = max(now, _RATE_LIMIT_NEXT_REQUESTS.get(key, now))
        _RATE_LIMIT_NEXT_REQUESTS[key] = request_at + interval
    delay = request_at - now
    if delay > 0:
        time.sleep(delay)


def _register_rate_limit(
    key: tuple[str, str],
    *,
    detail: str,
    retry_after: object,
) -> tuple[float, int]:
    limit = _rate_limit_from_detail(detail) or 10
    requests_per_minute = max(1, limit - 1)
    interval = 60.0 / requests_per_minute
    try:
        delay = float(retry_after) if isinstance(retry_after, str) else 60.0
    except ValueError:
        delay = 60.0
    delay = max(delay, interval)
    with _RATE_LIMIT_LOCK:
        now = time.monotonic()
        _RATE_LIMIT_INTERVALS[key] = max(
            interval,
            _RATE_LIMIT_INTERVALS.get(key, 0.0),
        )
        _RATE_LIMIT_NEXT_REQUESTS[key] = max(
            now + delay,
            _RATE_LIMIT_NEXT_REQUESTS.get(key, now),
        )
    return delay, requests_per_minute


def _rate_limit_from_detail(detail: str) -> int | None:
    try:
        value = cast(object, json.loads(detail))
    except json.JSONDecodeError:
        return None
    mapping = _string_mapping(value)
    if mapping is None:
        return None
    limit = mapping.get("limit")
    return limit if isinstance(limit, int) and not isinstance(limit, bool) else None


def _response_json(response: Mapping[str, object]) -> object:
    if response.get("status") != "completed":
        raise ValueError(
            f"Responses API did not complete: {response.get('incomplete_details')!r}"
        )
    outputs = _object_sequence(response.get("output"))
    texts: list[str] = []
    for output_value in outputs:
        output = _string_mapping(output_value)
        if output is None or output.get("type") != "message":
            continue
        for content_value in _object_sequence(output.get("content")):
            content = _string_mapping(content_value)
            if content is None:
                continue
            if content.get("type") == "refusal":
                raise ValueError(f"Responses API refusal: {content.get('refusal')}")
            if content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text)
    if not texts:
        raise ValueError("Responses API returned no non-empty output_text")
    return cast(object, json.loads("".join(texts)))


def _validation_retry_body(body: Mapping[str, object]) -> dict[str, object]:
    text = _string_mapping(body.get("text")) or {}
    output_format = _string_mapping(text.get("format")) or {}
    schema = _string_mapping(output_format.get("schema")) or {}
    required = [
        item
        for item in _object_sequence(schema.get("required"))
        if isinstance(item, str)
    ]
    required_text = ", ".join(required) if required else "all schema-required fields"
    instruction = (
        "Your previous output failed strict JSON schema validation. Return a complete "
        "JSON object with every required field and no text outside the JSON object. "
        f"Required fields: {required_text}. Do not omit fields; every required string "
        "must be non-empty."
    )
    retry_body = dict(body)
    retry_body["input"] = [
        *_object_sequence(body.get("input")),
        {
            "role": "user",
            "content": [{"type": "input_text", "text": instruction}],
        },
    ]
    return retry_body


def _response_diagnostic(response: Mapping[str, object]) -> str:
    output_types: list[str] = []
    content_types: list[str] = []
    text_lengths: list[int] = []
    for output_value in _object_sequence(response.get("output")):
        output = _string_mapping(output_value)
        if output is None:
            continue
        output_type = output.get("type")
        if isinstance(output_type, str):
            output_types.append(output_type)
        for content_value in _object_sequence(output.get("content")):
            content = _string_mapping(content_value)
            if content is None:
                continue
            content_type = content.get("type")
            if isinstance(content_type, str):
                content_types.append(content_type)
            text = content.get("text")
            if isinstance(text, str):
                text_lengths.append(len(text))
    return (
        "response metadata: "
        f"status={response.get('status')!r}, "
        f"output_types={output_types!r}, "
        f"content_types={content_types!r}, "
        f"text_lengths={text_lengths!r}, "
        f"incomplete_details={response.get('incomplete_details')!r}, "
        f"usage={response.get('usage')!r}"
    )


def _json_object(value: object) -> dict[str, object]:
    mapping = _string_mapping(value)
    if mapping is None:
        raise ValueError("JSON payload must be an object with string keys")
    return mapping


def _string_mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    mapping = cast(Mapping[object, object], value)
    output: dict[str, object] = {}
    for key, item in mapping.items():
        if not isinstance(key, str):
            return None
        output[key] = item
    return output


def _object_sequence(value: object) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return value


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ResponsesSettings",
    "StructuredResponse",
    "complete_structured_request",
    "load_responses_settings",
    "structured_request_body",
]

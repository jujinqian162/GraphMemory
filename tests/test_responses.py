from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import cast

import pytest
from typing_extensions import override

from graph_memory.infrastructure.responses import (
    ResponsesSettings,
    complete_structured_request,
    load_responses_settings,
    structured_request_body,
)


def test_load_responses_settings_prefers_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_path = tmp_path / ".env"
    _ = env_path.write_text(
        "MODEL_ID=file-model\nAPI_KEY=file-key\nBASE_URL=https://file.test/v1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MODEL_ID", "environment-model")

    settings = load_responses_settings(env_path)

    assert settings == ResponsesSettings(
        model_id="environment-model",
        api_key="file-key",
        base_url="https://file.test/v1",
    )


def test_complete_structured_request_reuses_valid_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = ResponsesSettings(
        model_id="model",
        api_key="secret",
        base_url="https://example.test/v1",
    )
    body = structured_request_body(
        settings=settings,
        system_prompt="system",
        user_payload={"query": "question"},
        schema_name="answer",
        schema={"type": "object"},
        max_output_tokens=32,
        prompt_cache_key="cache-key",
    )
    response = {
        "id": "response-id",
        "status": "completed",
        "usage": {"input_tokens": 10},
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps({"answer": "value"}),
                    }
                ],
            }
        ],
    }
    calls = 0

    class FakeResponse(BytesIO):
        @override
        def __enter__(self) -> FakeResponse:
            return super().__enter__()

        @override
        def __exit__(self, *args: object) -> None:
            super().__exit__(None, None, None)

    def fake_urlopen(request: object, *, timeout: float) -> FakeResponse:
        nonlocal calls
        del request, timeout
        calls += 1
        return FakeResponse(json.dumps(response).encode("utf-8"))

    monkeypatch.setattr(
        "graph_memory.infrastructure.responses.urllib.request.urlopen",
        fake_urlopen,
    )

    first = complete_structured_request(
        body,
        settings=settings,
        cache_dir=tmp_path / "cache",
        validate=_validate_answer,
    )
    second = complete_structured_request(
        body,
        settings=settings,
        cache_dir=tmp_path / "cache",
        validate=_validate_answer,
    )

    assert calls == 1
    assert first.cached is False
    assert second.cached is True
    assert first.request_digest == second.request_digest
    assert second.value == {"answer": "value"}


def test_complete_structured_request_retries_empty_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = ResponsesSettings(
        model_id="model",
        api_key="secret",
        base_url="https://example.test/v1",
    )
    body = structured_request_body(
        settings=settings,
        system_prompt="system",
        user_payload={"query": "question"},
        schema_name="answer",
        schema={"type": "object"},
        max_output_tokens=32,
        prompt_cache_key="cache-key",
    )
    responses = [
        _response(""),
        _response(json.dumps({"answer": "value"})),
    ]

    def fake_post_response(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        return responses.pop(0)

    monkeypatch.setattr(
        "graph_memory.infrastructure.responses._post_response",
        fake_post_response,
    )

    def no_sleep(seconds: float) -> None:
        del seconds

    monkeypatch.setattr("graph_memory.infrastructure.responses.time.sleep", no_sleep)

    result = complete_structured_request(
        body,
        settings=settings,
        cache_dir=tmp_path / "cache",
        validate=_validate_answer,
    )

    assert result.value == {"answer": "value"}
    assert responses == []


def _response(text: str) -> dict[str, object]:
    return {
        "id": "response-id",
        "status": "completed",
        "usage": {"output_tokens": 1},
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
    }


def _validate_answer(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("invalid answer")
    mapping = cast(dict[str, object], value)
    answer = mapping.get("answer")
    if not isinstance(answer, str):
        raise ValueError("invalid answer")
    return {"answer": answer}

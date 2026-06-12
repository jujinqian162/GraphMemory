from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import sentence_transformers

from graph_memory.embeddings.sentence_transformers import load_sentence_transformer


def test_load_sentence_transformer_resolves_existing_local_model_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name_or_path: str, **kwargs: object) -> None:
            captured["model_name_or_path"] = model_name_or_path
            captured["kwargs"] = kwargs

    local_model = tmp_path / "models" / "local-e5"
    local_model.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeSentenceTransformer)

    model = load_sentence_transformer("models/local-e5", device="cpu")

    assert isinstance(model, FakeSentenceTransformer)
    assert captured == {
        "model_name_or_path": str(local_model.resolve()),
        "kwargs": {"device": "cpu"},
    }


def test_load_sentence_transformer_preserves_hub_model_id(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name_or_path: str, **kwargs: object) -> None:
            captured["model_name_or_path"] = model_name_or_path
            captured["kwargs"] = kwargs

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeSentenceTransformer)

    model = load_sentence_transformer("intfloat/e5-base-v2")

    assert isinstance(model, FakeSentenceTransformer)
    assert captured == {
        "model_name_or_path": "intfloat/e5-base-v2",
        "kwargs": {},
    }


def test_sentence_transformer_construction_stays_in_embedding_loader() -> None:
    import graph_memory.models.dense_finetune.training as dense_ft_training
    import graph_memory.models.graph_retriever.text_embeddings as graph_text_embeddings
    import graph_memory.registry.retrieval_builders as retrieval_builders
    import graph_memory.retrieval.methods.flat.dense as flat_dense

    offenders = [
        module.__name__
        for module in (
            dense_ft_training,
            graph_text_embeddings,
            retrieval_builders,
            flat_dense,
        )
        if "SentenceTransformer(" in inspect.getsource(module)
    ]

    assert offenders == []

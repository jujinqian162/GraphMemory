from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_sentence_transformer_model_path(model_name_or_path: str | Path) -> str:
    value = str(model_name_or_path)
    path = Path(value)
    if path.exists():
        return str(path.resolve())
    return value


def load_sentence_transformer(model_name_or_path: str | Path, *, device: str | None = None) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError("sentence-transformers is required to load dense encoders.") from error
    resolved_model_name_or_path = resolve_sentence_transformer_model_path(model_name_or_path)
    if device is None:
        return SentenceTransformer(resolved_model_name_or_path)
    return SentenceTransformer(resolved_model_name_or_path, device=device)

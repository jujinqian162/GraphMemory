from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from graph_memory.retrieval.contracts import RankedNode, RetrievalMethodResult
from graph_memory.retrieval.requests import RankingMethodRequest, TextRankingRequest


class CrossEncoderScorer(Protocol):
    def predict(
        self,
        sentences: Sequence[Sequence[str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> object: ...


def load_cross_encoder(
    model_name_or_path: str | Path,
    *,
    device: str,
    max_length: int | None = None,
    num_labels: int | None = None,
) -> Any:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as error:
        raise RuntimeError(
            "sentence-transformers==2.7.0 is required to load cross encoders."
        ) from error
    value = str(model_name_or_path)
    path = Path(value)
    resolved = str(path.resolve()) if path.exists() else value
    if num_labels is not None:
        return (
            CrossEncoder(
                resolved,
                device=device,
                max_length=max_length,
                num_labels=num_labels,
                tokenizer_args={"fix_mistral_regex": False},
            )
            if max_length is not None
            else CrossEncoder(
                resolved,
                device=device,
                num_labels=num_labels,
                tokenizer_args={"fix_mistral_regex": False},
            )
        )
    return (
        CrossEncoder(
            resolved,
            device=device,
            max_length=max_length,
            tokenizer_args={"fix_mistral_regex": False},
        )
        if max_length is not None
        else CrossEncoder(
            resolved,
            device=device,
            tokenizer_args={"fix_mistral_regex": False},
        )
    )


class CrossEncoderTaskRetriever:
    name = "cross_encoder"

    def __init__(
        self,
        model: CrossEncoderScorer,
        *,
        batch_size: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("Cross-Encoder batch size must be positive")
        self.model = model
        self.batch_size = batch_size

    def rank_task(
        self, request: RankingMethodRequest, *, top_k: int
    ) -> RetrievalMethodResult:
        del top_k
        if not isinstance(request, TextRankingRequest):
            raise TypeError(
                "cross_encoder requires TextRankingRequest, "
                f"got {type(request).__name__}."
            )
        pairs = [[request.query_text, candidate.text] for candidate in request.candidates]
        raw_scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = np.asarray(raw_scores, dtype=float).reshape(-1)
        if scores.shape != (len(request.candidates),):
            raise ValueError(
                "Cross-Encoder returned invalid score shape: "
                f"expected={(len(request.candidates),)} observed={scores.shape}"
            )
        if not np.isfinite(scores).all():
            raise ValueError("Cross-Encoder returned non-finite scores")
        ranked = [
            RankedNode(node_id=candidate.item_id, score=float(score))
            for candidate, score in zip(request.candidates, scores, strict=True)
        ]
        return RetrievalMethodResult(
            ranked_nodes=tuple(
                sorted(ranked, key=lambda row: (-row.score, row.node_id))
            )
        )


__all__ = [
    "CrossEncoderScorer",
    "CrossEncoderTaskRetriever",
    "load_cross_encoder",
]

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from graph_memory.embeddings import SentenceEncoder
from graph_memory.retrieval.requests import (
    GraphRAGRequest,
    GraphRAGResolverEvidence,
    GraphRAGTitleEntityGroup,
    TextCandidate,
)


@dataclass(frozen=True)
class SentenceResolutionInput:
    anchor_candidate_id: str
    title_group: GraphRAGTitleEntityGroup


@dataclass(frozen=True)
class GraphRAGSentenceResolver:
    encoder: SentenceEncoder
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    batch_size: int = 64
    min_score_margin: float = 0.02

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("GraphRAG resolver batch_size must be positive.")
        if not math.isfinite(self.min_score_margin) or self.min_score_margin < 0.0:
            raise ValueError(
                "GraphRAG resolver min_score_margin must be finite/non-negative."
            )

    def resolve_many(
        self,
        request: GraphRAGRequest,
        inputs: Sequence[SentenceResolutionInput],
        *,
        original_dense_rank: dict[str, int],
    ) -> tuple[GraphRAGResolverEvidence, ...]:
        ordered_inputs = tuple(inputs)
        if not ordered_inputs:
            return ()
        candidate_by_id = {
            candidate.item_id: candidate for candidate in request.candidates
        }
        anchor_ids = tuple(
            dict.fromkeys(item.anchor_candidate_id for item in ordered_inputs)
        )
        passage_ids = tuple(
            dict.fromkeys(
                candidate_id
                for item in ordered_inputs
                for candidate_id in item.title_group.candidate_ids
                if candidate_id != item.anchor_candidate_id
            )
        )
        query_texts = [
            self.query_prefix
            + request.query_text
            + "\nSource evidence: "
            + _candidate_evidence_text(candidate_by_id[anchor_id])
            for anchor_id in anchor_ids
        ]
        passage_texts = [
            self.passage_prefix
            + _candidate_evidence_text(candidate_by_id[candidate_id])
            for candidate_id in passage_ids
        ]
        matrix = _encode(
            self.encoder,
            [*query_texts, *passage_texts],
            batch_size=self.batch_size,
        )
        query_by_anchor = {
            anchor_id: matrix[index] for index, anchor_id in enumerate(anchor_ids)
        }
        passage_offset = len(anchor_ids)
        passage_by_id = {
            candidate_id: matrix[passage_offset + index]
            for index, candidate_id in enumerate(passage_ids)
        }

        results: list[GraphRAGResolverEvidence] = []
        for item in ordered_inputs:
            target_ids = tuple(
                candidate_id
                for candidate_id in item.title_group.candidate_ids
                if candidate_id != item.anchor_candidate_id
            )
            if not target_ids:
                results.append(
                    GraphRAGResolverEvidence(
                        anchor_candidate_id=item.anchor_candidate_id,
                        entity_id=item.title_group.entity_id,
                        candidate_ids=item.title_group.candidate_ids,
                        selected_candidate_id=None,
                        top1_score=None,
                        top2_score=None,
                        score_margin=None,
                        accepted=False,
                        rejection_reason="self_target_only",
                    )
                )
                continue
            query_vector = query_by_anchor[item.anchor_candidate_id]
            scored = sorted(
                (
                    (
                        candidate_id,
                        float(query_vector @ passage_by_id[candidate_id]),
                    )
                    for candidate_id in target_ids
                ),
                key=lambda row: (
                    -row[1],
                    original_dense_rank[row[0]],
                    row[0],
                ),
            )
            top1_id, top1_score = scored[0]
            top2_score = scored[1][1] if len(scored) > 1 else None
            margin = top1_score - top2_score if top2_score is not None else None
            ambiguous = margin is not None and margin < self.min_score_margin
            results.append(
                GraphRAGResolverEvidence(
                    anchor_candidate_id=item.anchor_candidate_id,
                    entity_id=item.title_group.entity_id,
                    candidate_ids=item.title_group.candidate_ids,
                    selected_candidate_id=None if ambiguous else top1_id,
                    top1_score=top1_score,
                    top2_score=top2_score,
                    score_margin=margin,
                    accepted=not ambiguous,
                    rejection_reason=(
                        "ambiguous_title_sentence" if ambiguous else None
                    ),
                )
            )
        return tuple(results)


def _candidate_evidence_text(candidate: TextCandidate) -> str:
    title_value = candidate.metadata.get("title")
    title = title_value if isinstance(title_value, str) else ""
    text = candidate.text
    if title:
        prefix = f"{title}. "
        if text.casefold().startswith(prefix.casefold()):
            text = text[len(prefix) :]
        return f"{title}. {text}"
    return text


def _encode(
    encoder: SentenceEncoder,
    texts: Sequence[str],
    *,
    batch_size: int,
) -> np.ndarray:
    value = encoder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
    )
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != len(texts) or matrix.shape[1] <= 0:
        raise ValueError(
            "GraphRAG resolver returned invalid embeddings: "
            f"expected_rows={len(texts)} observed_shape={matrix.shape}."
        )
    return matrix


__all__ = ["GraphRAGSentenceResolver", "SentenceResolutionInput"]

"""Query-conditioned relation semantics for typed partner completion.

The frozen retriever encoder scores the request query against schema-owned
natural-language descriptions of each provenance edge type. The resulting
softmax is the only typed signal the retriever uses: there are no hand-authored
per-type priors and no schema gates, so the same code generalizes across graph
families without reading dataset identity or labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from graph_memory.embeddings import SentenceEncoder
from graph_memory.retrieval.methods.epgm.config import EpgmRetrieverConfig


@dataclass(frozen=True)
class RelationAffinity:
    """Query compatibility and graph specificity for one edge type.

    ``affinity`` is the softmax distribution over the traversable edge types
    present in the graph and is what the audit trace reports. ``preference``
    rescales it so the best-matching relation is exactly 1.0. ``specificity``
    is an inverse-frequency weight computed from the graph's own edge-type
    counts.

    Traversal uses ``preference * specificity``, not ``affinity``. Two reasons:

    * A distribution sums to one, so each entry shrinks as the number of present
      edge types grows. Using it as a multiplicative path factor would make
      confidence depend on how many relation types a graph happens to contain
      rather than on how well a path matches the query.
    * Query-relation similarity alone favours whichever relation is most
      generic, because generic relations describe most queries tolerably well.
      On real agent traces the bulk relation (`depends_on`) is the least
      informative one, while rare relations (`invalidates`, `contradicts`,
      `verifies`) are the ones that actually identify evidence. Specificity is
      the standard inverse-frequency correction and is computed per graph from
      structure alone, with no labels and no tuned constant.
    """

    edge_type: str
    similarity: float
    affinity: float
    preference: float
    specificity: float

    @property
    def traversal_weight(self) -> float:
        """Query match times graph specificity; the factor a walk multiplies."""
        return self.preference * self.specificity


def encode_relation_vectors(
    encoder: SentenceEncoder,
    config: EpgmRetrieverConfig,
    *,
    passage_prefix: str,
    batch_size: int,
) -> NDArray[np.float64]:
    edge_types = sorted(config.relation_descriptions)
    texts = [passage_prefix + config.relation_descriptions[edge_type] for edge_type in edge_types]
    matrix = np.asarray(
        encoder.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=float,
    )
    if matrix.ndim != 2 or matrix.shape[0] != len(edge_types) or matrix.shape[1] <= 0:
        raise ValueError("relation encoder returned an invalid embedding matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("relation encoder returned non-finite embeddings")
    return matrix


def encode_query_vector(
    encoder: SentenceEncoder,
    query_text: str,
    *,
    query_prefix: str,
    batch_size: int,
) -> NDArray[np.float64]:
    matrix = np.asarray(
        encoder.encode(
            [query_prefix + query_text],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=float,
    )
    if matrix.ndim != 2 or matrix.shape[0] != 1 or matrix.shape[1] <= 0:
        raise ValueError("query encoder returned an invalid embedding matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("query encoder returned non-finite embeddings")
    return matrix[0]


def query_relation_affinities(
    query_vector: NDArray[np.float64],
    relation_vectors: NDArray[np.float64],
    present_edge_types: frozenset[str],
    config: EpgmRetrieverConfig,
    *,
    edge_type_counts: Mapping[str, int] | None = None,
) -> tuple[RelationAffinity, ...]:
    """Softmax query/relation cosine, with per-graph inverse-frequency weights.

    ``edge_type_counts`` supplies how often each traversable type occurs in this
    graph. When omitted every specificity is 1.0, which isolates the pure
    query-similarity signal for ablation.
    """

    edge_types = sorted(config.relation_descriptions)
    if relation_vectors.shape != (len(edge_types), query_vector.shape[0]):
        raise ValueError("query and relation embedding dimensions do not match")
    present = [edge_type for edge_type in edge_types if edge_type in present_edge_types]
    if not present:
        return ()
    index_by_type = {edge_type: index for index, edge_type in enumerate(edge_types)}
    similarities = np.asarray(
        [float(relation_vectors[index_by_type[edge_type]] @ query_vector) for edge_type in present],
        dtype=float,
    )
    logits = similarities / config.relation_temperature
    logits -= float(np.max(logits))
    weights = np.exp(logits)
    weights /= float(np.sum(weights))
    peak = float(np.max(weights))
    preferences = weights / peak if peak > 0.0 else np.ones_like(weights)
    specificities = _relation_specificity(present, edge_type_counts)
    return tuple(
        RelationAffinity(
            edge_type,
            float(similarity),
            float(affinity),
            float(preference),
            float(specificity),
        )
        for edge_type, similarity, affinity, preference, specificity in zip(
            present, similarities, weights, preferences, specificities, strict=True
        )
    )


def _relation_specificity(
    present: Sequence[str],
    edge_type_counts: Mapping[str, int] | None,
) -> NDArray[np.float64]:
    """Inverse-frequency weight per relation, peak-normalized to 1.0.

    ``log(1 + total / count)`` is the usual IDF form. Peak normalization keeps
    the factor scale-free so one confidence threshold stays meaningful whether a
    graph has three relation types or thirteen.
    """

    if edge_type_counts is None:
        return np.ones(len(present), dtype=float)
    counts = np.asarray(
        [max(int(edge_type_counts.get(edge_type, 0)), 0) for edge_type in present],
        dtype=float,
    )
    total = float(counts.sum())
    if total <= 0.0:
        return np.ones(len(present), dtype=float)
    safe = np.where(counts > 0.0, counts, total)
    values = np.log1p(total / safe)
    peak = float(np.max(values))
    if peak <= 0.0:
        return np.ones(len(present), dtype=float)
    return values / peak


__all__ = [
    "RelationAffinity",
    "encode_query_vector",
    "encode_relation_vectors",
    "query_relation_affinities",
]


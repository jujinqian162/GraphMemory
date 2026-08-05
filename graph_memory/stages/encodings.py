from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from numpy.lib.format import open_memmap
from numpy.typing import NDArray
from pydantic import TypeAdapter

from graph_memory.datasets.isetrace.benchmark_records import ISETraceRankingRecord
from graph_memory.datasets.selection import text_ranking_requests_for_dataset
from graph_memory.embeddings import (
    format_dense_passage,
    format_dense_query,
    load_sentence_transformer,
)
from graph_memory.experiment.artifacts import (
    ArtifactKind,
    ArtifactPublisher,
    DatasetArtifactRef,
    DirectorySourceRef,
    EvidenceGraphArtifactRef,
    FileSourceRef,
    FrozenEmbeddingsArtifactRef,
    ModelArtifactRef,
    ProcessedAssetStore,
    RevisionSourceRef,
    artifact_payload_path,
)
from graph_memory.experiment.config import DatasetName, DenseEncoderConfig
from graph_memory.graphs.contracts import EvidenceGraph
from graph_memory.graphs.provenance import ProvenanceGraph
from graph_memory.io import read_json, write_json
from graph_memory.models.frozen_embeddings import (
    EmbeddingSplit,
    FrozenEmbeddingGroup,
    FrozenEmbeddingIndex,
    node_ids_digest,
)
from graph_memory.models.graph_retriever.provenance import (
    provenance_embedding_request,
)
from graph_memory.retrieval.requests import ProvenanceRgcnRequest
from graph_memory.stages.results import FrozenEmbeddingsResult


EncoderSourceRef = FileSourceRef | DirectorySourceRef | RevisionSourceRef
EVIDENCE_GRAPHS_ADAPTER = TypeAdapter(list[EvidenceGraph])
PROVENANCE_GRAPHS_ADAPTER = TypeAdapter(list[ProvenanceGraph])
ISETRACE_RANKINGS_ADAPTER = TypeAdapter(list[ISETraceRankingRecord])
EncodingFamily = Literal["evidence", "provenance"]


@dataclass(frozen=True)
class _TextGroup:
    split: EmbeddingSplit
    task_id: str
    node_ids: tuple[str, ...]
    texts: tuple[str, ...]


def materialize_frozen_rgcn_embeddings(
    store: ProcessedAssetStore,
    *,
    dataset: DatasetName,
    encoder: DenseEncoderConfig,
    encoder_source: EncoderSourceRef,
    train_prepared: DatasetArtifactRef,
    dev_prepared: DatasetArtifactRef,
    train_graphs: EvidenceGraphArtifactRef | None,
    dev_graphs: EvidenceGraphArtifactRef | None,
    seed_model: ModelArtifactRef | None,
    enable_gpupool: bool,
    device: str,
    chunk_size: int,
    implementation_version: str,
    sentence_transformer: Any | None = None,
) -> FrozenEmbeddingsResult:
    family: EncodingFamily = (
        "provenance" if dataset == "isetrace" else "evidence"
    )
    groups = [
        *_groups_for_split(
            family,
            "train",
            dataset=dataset,
            prepared=train_prepared,
            graphs=train_graphs,
            encoder=encoder,
        ),
        *_groups_for_split(
            family,
            "dev",
            dataset=dataset,
            prepared=dev_prepared,
            graphs=dev_graphs,
            encoder=encoder,
        ),
    ]
    if not groups:
        raise ValueError("Frozen R-GCN encoding requires at least one task.")
    texts = [text for group in groups for text in group.texts]
    model_name = _effective_model_name(encoder, encoder_source, seed_model)
    pool_devices = resolve_encoding_devices(enable_gpupool, device)
    model = sentence_transformer or load_sentence_transformer(
        model_name,
        device=pool_devices[0],
    )

    with ArtifactPublisher(
        store,
        kind=ArtifactKind.FROZEN_EMBEDDINGS,
        namespace=dataset,
        task_identity=f"encode-{family}-rgcn",
        origin={
            "stage": "encode",
            "family": family,
            "dataset": dataset,
            "train_prepared_digest": train_prepared.digest,
            "dev_prepared_digest": dev_prepared.digest,
            "train_graph_digest": None if train_graphs is None else train_graphs.digest,
            "dev_graph_digest": None if dev_graphs is None else dev_graphs.digest,
            "encoder_identity": _encoder_identity(encoder_source),
            "seed_model_digest": None if seed_model is None else seed_model.digest,
            "query_prefix": encoder.query_prefix,
            "passage_prefix": encoder.passage_prefix,
            "encoder_batch_size": encoder.batch_size,
            "implementation_version": implementation_version,
        },
    ) as publisher:
        embedding_dim = write_sentence_transformer_embeddings(
            model,
            texts,
            output=publisher.workspace / "embeddings.npy",
            pool_devices=pool_devices,
            batch_size=encoder.batch_size,
            chunk_size=chunk_size,
        )
        index_groups: list[FrozenEmbeddingGroup] = []
        row_start = 0
        for group in groups:
            row_end = row_start + len(group.texts)
            index_groups.append(
                FrozenEmbeddingGroup(
                    split=group.split,
                    task_id=group.task_id,
                    row_start=row_start,
                    row_end=row_end,
                    node_ids_digest=node_ids_digest(group.node_ids),
                )
            )
            row_start = row_end
        index = FrozenEmbeddingIndex(
            family=family,
            embedding_dim=embedding_dim,
            row_count=len(texts),
            groups=tuple(index_groups),
        )
        write_json(
            publisher.workspace / "index.json",
            index.model_dump(mode="json"),
        )
        artifact = publisher.publish(
            {"embeddings": "embeddings.npy", "index": "index.json"},
            shape={
                "rows": len(texts),
                "embedding_dim": embedding_dim,
                "tasks": len(groups),
            },
            metadata={"family": family},
        )
    assert isinstance(artifact, FrozenEmbeddingsArtifactRef)
    return FrozenEmbeddingsResult(artifact=artifact)


def resolve_encoding_devices(enable_gpupool: bool, device: str) -> tuple[str, ...]:
    if not enable_gpupool:
        return (device,)
    device_count = torch.cuda.device_count()
    if device_count <= 0:
        raise RuntimeError(
            "encoding.enable_gpupool=true requires at least one CUDA device "
            "visible to PyTorch."
        )
    return tuple(f"cuda:{index}" for index in range(device_count))


def write_sentence_transformer_embeddings(
    model: Any,
    texts: Sequence[str],
    *,
    output: Path,
    pool_devices: tuple[str, ...],
    batch_size: int,
    chunk_size: int,
) -> int:
    """Stream SentenceTransformers output into one float32 .npy matrix."""

    if not texts:
        raise ValueError("Cannot encode an empty text sequence.")
    if batch_size <= 0 or chunk_size <= 0:
        raise ValueError("Encoding batch_size and chunk_size must be positive.")
    getter = getattr(model, "get_sentence_embedding_dimension", None)
    dimension = getter() if callable(getter) else None
    if not isinstance(dimension, int) or dimension <= 0:
        probe = _encode_chunk(
            model,
            [texts[0]],
            pool=None,
            batch_size=1,
        )
        dimension = int(probe.shape[1])

    matrix = open_memmap(
        output,
        mode="w+",
        dtype=np.float32,
        shape=(len(texts), dimension),
    )
    pool: dict[str, object] | None = None
    try:
        if len(pool_devices) > 1:
            pool = cast(
                dict[str, object],
                model.start_multi_process_pool(target_devices=list(pool_devices)),
            )
        for row_start in range(0, len(texts), chunk_size):
            row_end = min(len(texts), row_start + chunk_size)
            encoded = _encode_chunk(
                model,
                list(texts[row_start:row_end]),
                pool=pool,
                batch_size=batch_size,
            )
            if encoded.shape != (row_end - row_start, dimension):
                raise ValueError(
                    "SentenceTransformer returned an invalid embedding shape: "
                    f"expected={(row_end - row_start, dimension)} "
                    f"observed={encoded.shape}."
                )
            matrix[row_start:row_end] = encoded
        matrix.flush()
    finally:
        if pool is not None:
            model.stop_multi_process_pool(pool)
    return dimension


def _encode_chunk(
    model: Any,
    texts: list[str],
    *,
    pool: dict[str, object] | None,
    batch_size: int,
) -> NDArray[np.float32]:
    if pool is None:
        value = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    else:
        value = model.encode_multi_process(
            texts,
            pool,
            batch_size=batch_size,
            normalize_embeddings=True,
        )
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(texts) or matrix.shape[1] <= 0:
        raise ValueError(
            "SentenceTransformer returned an invalid embedding matrix: "
            f"texts={len(texts)} observed={matrix.shape}."
        )
    return matrix


def _groups_for_split(
    family: EncodingFamily,
    split: EmbeddingSplit,
    *,
    dataset: DatasetName,
    prepared: DatasetArtifactRef,
    graphs: EvidenceGraphArtifactRef | None,
    encoder: DenseEncoderConfig,
) -> list[_TextGroup]:
    task_inputs = cast(
        list[object], read_json(artifact_payload_path(prepared, "tasks"))
    )
    if family == "provenance":
        rankings = ISETRACE_RANKINGS_ADAPTER.validate_python(task_inputs)
        graph_values = PROVENANCE_GRAPHS_ADAPTER.validate_python(
            read_json(artifact_payload_path(prepared, "provenance_graphs"))
        )
        graphs_by_id = {graph.graph_id: graph for graph in graph_values}
        provenance_groups: list[_TextGroup] = []
        for ranking in rankings:
            request = ProvenanceRgcnRequest(
                task_id=ranking.task_id,
                query_text=ranking.query_text,
                candidates=ranking.provenance_candidates,
                graph=graphs_by_id[ranking.graph_id],
            )
            embedding_request, node_ids = provenance_embedding_request(request)
            text_by_node_id = {
                "q": format_dense_query(
                    embedding_request, query_prefix=encoder.query_prefix
                ),
                **{
                    candidate.item_id: format_dense_passage(
                        candidate, passage_prefix=encoder.passage_prefix
                    )
                    for candidate in embedding_request.candidates
                },
            }
            provenance_groups.append(
                _TextGroup(
                    split=split,
                    task_id=request.task_id,
                    node_ids=tuple(node_ids),
                    texts=tuple(text_by_node_id[node_id] for node_id in node_ids),
                )
            )
        return provenance_groups
    if graphs is None:
        raise ValueError("Evidence frozen encoding requires evidence graphs.")
    graph_values = EVIDENCE_GRAPHS_ADAPTER.validate_python(
        read_json(artifact_payload_path(graphs, "graphs"))
    )
    graphs_by_task = {graph.task_id: graph for graph in graph_values}
    groups: list[_TextGroup] = []
    for request in text_ranking_requests_for_dataset(dataset, task_inputs):
        graph = graphs_by_task[request.task_id]
        node_ids = tuple(node.id for node in graph.nodes)
        text_by_node_id = {
            "q": format_dense_query(request, query_prefix=encoder.query_prefix),
            **{
                candidate.item_id: format_dense_passage(
                    candidate, passage_prefix=encoder.passage_prefix
                )
                for candidate in request.candidates
            },
        }
        try:
            texts = tuple(text_by_node_id[node_id] for node_id in node_ids)
        except KeyError as error:
            raise ValueError(
                "Evidence graph contains a node absent from its text request: "
                f"task_id={request.task_id!r} node_id={error.args[0]!r}."
            ) from error
        groups.append(
            _TextGroup(
                split=split,
                task_id=request.task_id,
                node_ids=node_ids,
                texts=texts,
            )
        )
    return groups


def _effective_model_name(
    encoder: DenseEncoderConfig,
    encoder_source: EncoderSourceRef,
    seed_model: ModelArtifactRef | None,
) -> str:
    if seed_model is not None:
        return artifact_payload_path(seed_model, "model").as_posix()
    if isinstance(encoder_source, (FileSourceRef, DirectorySourceRef)):
        return encoder_source.uri
    return encoder.model_name


def _encoder_identity(source: EncoderSourceRef) -> str:
    if isinstance(source, (FileSourceRef, DirectorySourceRef)):
        return source.digest
    return f"{source.uri}@{source.revision}"


__all__ = [
    "EncoderSourceRef",
    "materialize_frozen_rgcn_embeddings",
    "resolve_encoding_devices",
    "write_sentence_transformer_embeddings",
]

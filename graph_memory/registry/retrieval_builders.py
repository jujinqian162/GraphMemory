from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

from typing_extensions import NotRequired

from graph_memory.contracts.common import JsonValue as RecursiveJsonValue

from graph_memory.embeddings import SentenceEncoder, load_sentence_transformer
from graph_memory.experiment.config import (
    Bm25MethodConfig,
    CrossEncoderMethodConfig,
    DenseCandidateView,
    DenseEncoderConfig,
    DenseFinetuneMethodConfig,
    DenseFtRgcnMethodConfig,
    DenseMethodConfig,
    GraphRAGMethodConfig,
    MethodConfig,
    ProvenancePathMethodConfig,
    ProvenanceRgcnMethodConfig,
    RgcnMethodConfig,
)
from graph_memory.graphs.contracts import EvidenceGraph
from graph_memory.graphs.index import GraphIndex
from graph_memory.graphs.provenance import ProvenanceGraph
from graph_memory.models.cross_encoder.metadata import load_cross_encoder_model_metadata
from graph_memory.models.dense_finetune.metadata import load_dense_ft_model_metadata
from graph_memory.models.graph_retriever.checkpoint import RgcnCheckpoint
from graph_memory.retrieval.contracts import RetrievalMethod
from graph_memory.retrieval.methods.flat.bm25 import BM25TaskRetriever
from graph_memory.retrieval.methods.flat.dense import DenseConfig, DenseTaskRetriever
from graph_memory.retrieval.methods.graphrag import (
    GraphRAGMethod,
    build_graphrag_knowledge_graph,
    build_graphrag_request,
)
from graph_memory.retrieval.methods.ids import RetrievalMethodId
from graph_memory.retrieval.requests import (
    EvidenceGraphRankingRequest,
    ExecutionProvenanceRankingRequest,
    GraphRAGKnowledgeGraph,
    RankingMethodRequest,
    TextRankingRequest,
)
from graph_memory.retrieval.signals import SeedSignalProvider

if TYPE_CHECKING:
    from graph_memory.models.graph_retriever.contracts import TextEmbeddingProvider


class RetrievalProvenance(TypedDict):
    method: str
    model: str | None
    device: str | None
    encoder: dict[str, RecursiveJsonValue] | None
    control: NotRequired[dict[str, RecursiveJsonValue]]


def build_retrieval(
    method_config: MethodConfig,
    *,
    text_requests: list[TextRankingRequest],
    device: str,
    encoder_model_name: str | None = None,
    checkpoint: Path | None = None,
    evidence_graphs: list[EvidenceGraph] | None = None,
    provenance_graphs: list[ProvenanceGraph] | None = None,
    graph_ids_by_task_id: Mapping[str, str] | None = None,
    dense_encoder: SentenceEncoder | None = None,
    text_embedding_provider: TextEmbeddingProvider | None = None,
    seed_signal_provider: SeedSignalProvider | None = None,
) -> tuple[RetrievalMethod, RetrievalProvenance, list[RankingMethodRequest]]:
    if isinstance(method_config, Bm25MethodConfig):
        return _built(
            BM25TaskRetriever(),
            method=RetrievalMethodId.BM25,
            execution_requests=list(text_requests),
        )
    if isinstance(method_config, DenseMethodConfig):
        encoder = _encoder_settings(method_config.encoder, encoder_model_name)
        return _built(
            _build_dense_ranker(encoder, dense_encoder, device=device),
            method=RetrievalMethodId.DENSE,
            device=device,
            encoder=encoder,
            execution_requests=list(text_requests),
        )
    if isinstance(method_config, DenseFinetuneMethodConfig):
        return _build_dense_ft(
            _required_checkpoint(checkpoint, method_config.method),
            method_config.variant,
            text_requests,
            dense_encoder=dense_encoder,
            device=device,
        )
    if isinstance(method_config, CrossEncoderMethodConfig):
        return _build_cross_encoder(
            _required_checkpoint(checkpoint, method_config.method),
            method_config,
            text_requests,
            device=device,
        )
    if isinstance(method_config, GraphRAGMethodConfig):
        return _build_graphrag(
            method_config,
            text_requests,
            encoder_model_name=encoder_model_name,
            dense_encoder=dense_encoder,
            device=device,
        )
    if isinstance(method_config, ProvenancePathMethodConfig):
        return _build_provenance_path(
            method_config,
            text_requests,
            provenance_graphs or [],
            graph_ids_by_task_id or {},
            encoder_model_name=encoder_model_name,
            dense_encoder=dense_encoder,
            device=device,
        )
    if isinstance(method_config, ProvenanceRgcnMethodConfig):
        return _build_provenance_rgcn(
            _required_checkpoint(checkpoint, method_config.method),
            method_config,
            text_requests,
            provenance_graphs or [],
            graph_ids_by_task_id or {},
            dense_encoder=dense_encoder,
            text_embedding_provider=text_embedding_provider,
            seed_signal_provider=seed_signal_provider,
            device=device,
        )
    if isinstance(method_config, (RgcnMethodConfig, DenseFtRgcnMethodConfig)):
        method = (
            RetrievalMethodId.DENSE_FT_RGCN_GRAPH_RETRIEVER
            if isinstance(method_config, DenseFtRgcnMethodConfig)
            else RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER
        )
        return _build_evidence_rgcn(
            _required_checkpoint(checkpoint, method_config.method),
            method,
            text_requests,
            evidence_graphs or [],
            dense_encoder=dense_encoder,
            text_embedding_provider=text_embedding_provider,
            seed_signal_provider=seed_signal_provider,
            device=device,
        )
    raise TypeError(f"unsupported method config={type(method_config).__name__}")


def _build_dense_ft(
    checkpoint: Path,
    variant: DenseCandidateView,
    text_requests: list[TextRankingRequest],
    *,
    dense_encoder: SentenceEncoder | None,
    device: str,
) -> tuple[RetrievalMethod, RetrievalProvenance, list[RankingMethodRequest]]:
    metadata = load_dense_ft_model_metadata(checkpoint)
    if metadata.variant != variant:
        checkpoint_variant = metadata.variant
        raise ValueError(
            f"Dense-FT checkpoint variant={checkpoint_variant!r} does not match requested variant={variant!r}."
        )
    encoder = dense_encoder
    if encoder is None:
        try:
            encoder = cast(
                SentenceEncoder,
                cast(object, load_sentence_transformer(checkpoint, device=device)),
            )
        except RuntimeError as error:
            raise RuntimeError(
                "sentence-transformers is required for dense-ft retrieval."
            ) from error
    encoder_settings = DenseEncoderConfig(
        model_name=metadata.base_model,
        query_prefix=metadata.query_prefix,
        passage_prefix=metadata.passage_prefix,
        batch_size=metadata.batch_size,
    )
    return _built(
        DenseTaskRetriever(
            config=DenseConfig(
                device=device,
                model_name=str(checkpoint),
                query_prefix=metadata.query_prefix,
                passage_prefix=metadata.passage_prefix,
                batch_size=metadata.batch_size,
            ),
            encoder=encoder,
            device=device,
            method_name=RetrievalMethodId.DENSE_FT.value,
        ),
        method=RetrievalMethodId.DENSE_FT,
        model=checkpoint,
        device=device,
        encoder=encoder_settings,
        execution_requests=list(text_requests),
    )


def _build_cross_encoder(
    checkpoint: Path,
    config: CrossEncoderMethodConfig,
    text_requests: list[TextRankingRequest],
    *,
    device: str,
) -> tuple[RetrievalMethod, RetrievalProvenance, list[RankingMethodRequest]]:
    from graph_memory.retrieval.methods.flat.cross_encoder import (
        CrossEncoderTaskRetriever,
        load_cross_encoder,
    )

    metadata = load_cross_encoder_model_metadata(checkpoint)
    if metadata.variant != config.variant:
        raise ValueError(
            f"Cross-Encoder checkpoint variant={metadata.variant!r} does not match "
            f"requested variant={config.variant!r}."
        )
    model = load_cross_encoder(
        checkpoint,
        device=device,
        max_length=metadata.max_length,
    )
    return _built(
        CrossEncoderTaskRetriever(model, batch_size=metadata.eval_batch_size),
        method=RetrievalMethodId.CROSS_ENCODER,
        model=checkpoint,
        device=device,
        encoder=config.backbone,
        execution_requests=list(text_requests),
    )


def _build_graphrag(
    config: GraphRAGMethodConfig,
    text_requests: list[TextRankingRequest],
    *,
    encoder_model_name: str | None,
    dense_encoder: SentenceEncoder | None,
    device: str,
) -> tuple[RetrievalMethod, RetrievalProvenance, list[RankingMethodRequest]]:
    encoder = _encoder_settings(config.encoder, encoder_model_name)
    dense_ranker = _build_dense_ranker(encoder, dense_encoder, device=device)
    graph_by_candidate_ids: dict[tuple[str, ...], GraphRAGKnowledgeGraph] = {}
    execution_requests: list[RankingMethodRequest] = []
    for request in text_requests:
        candidate_ids = tuple(candidate.item_id for candidate in request.candidates)
        graph = graph_by_candidate_ids.get(candidate_ids)
        if graph is None:
            graph = build_graphrag_knowledge_graph(request.candidates, config=config)
            graph_by_candidate_ids[candidate_ids] = graph
        execution_requests.append(
            build_graphrag_request(request, config, knowledge_graph=graph)
        )
    return _built(
        GraphRAGMethod(dense_ranker=dense_ranker, config=config),
        method=RetrievalMethodId.GRAPHRAG,
        device=device,
        encoder=encoder,
        execution_requests=execution_requests,
    )


def _build_provenance_path(
    config: ProvenancePathMethodConfig,
    text_requests: list[TextRankingRequest],
    graphs: list[ProvenanceGraph],
    graph_ids_by_task_id: Mapping[str, str],
    *,
    encoder_model_name: str | None,
    dense_encoder: SentenceEncoder | None,
    device: str,
) -> tuple[RetrievalMethod, RetrievalProvenance, list[RankingMethodRequest]]:
    from graph_memory.retrieval.methods.provenance_path import ProvenancePathMethod

    encoder = _encoder_settings(config.encoder, encoder_model_name)
    return _built(
        ProvenancePathMethod(
            dense_ranker=_build_dense_ranker(encoder, dense_encoder, device=device),
            config=config.retrieval_config(),
        ),
        method=RetrievalMethodId.PROVENANCE_PATH,
        device=device,
        encoder=encoder,
        execution_requests=_provenance_requests(
            text_requests,
            graphs,
            graph_ids_by_task_id,
            method="provenance path",
        ),
    )


def _build_provenance_rgcn(
    checkpoint_path: Path,
    method_config: ProvenanceRgcnMethodConfig,
    text_requests: list[TextRankingRequest],
    graphs: list[ProvenanceGraph],
    graph_ids_by_task_id: Mapping[str, str],
    *,
    dense_encoder: SentenceEncoder | None,
    text_embedding_provider: TextEmbeddingProvider | None,
    seed_signal_provider: SeedSignalProvider | None,
    device: str,
) -> tuple[RetrievalMethod, RetrievalProvenance, list[RankingMethodRequest]]:
    from graph_memory.models.graph_retriever.checkpoint import load_rgcn_checkpoint
    from graph_memory.models.graph_retriever.text_embeddings import (
        DenseGraphFeatureProvider,
    )
    from graph_memory.retrieval.methods.trainable_graph import (
        ProvenanceRgcnRetrievalMethod,
    )

    checkpoint = load_rgcn_checkpoint(
        checkpoint_path,
        expected_method=RetrievalMethodId.PROVENANCE_RGCN,
        map_location=device,
    )
    if checkpoint.model_config.ablation_name != method_config.variant:
        raise ValueError(
            "provenance R-GCN checkpoint variant does not match requested variant: "
            f"checkpoint={checkpoint.model_config.ablation_name!r} "
            f"requested={method_config.variant!r}"
        )
    provider = text_embedding_provider or DenseGraphFeatureProvider(
        model_name=checkpoint.model_config.encoder_model,
        query_prefix=checkpoint.model_config.query_prefix,
        passage_prefix=checkpoint.model_config.passage_prefix,
        batch_size=checkpoint.model_config.encoder_batch_size,
        device=device,
        encoder=dense_encoder,
    )
    if seed_signal_provider is None:
        if isinstance(provider, SeedSignalProvider):
            seed_signal_provider = provider
        else:
            raise ValueError(
                "provenance R-GCN requires an explicit seed signal provider when its text provider cannot score candidates"
            )
    encoder = _checkpoint_encoder(checkpoint)
    execution_requests = _provenance_requests(
        text_requests,
        graphs,
        graph_ids_by_task_id,
        method="provenance R-GCN",
    )
    built = _built(
        ProvenanceRgcnRetrievalMethod.from_checkpoint(
            checkpoint_path,
            text_embedding_provider=provider,
            seed_signal_provider=seed_signal_provider,
            device=device,
        ),
        method=RetrievalMethodId.PROVENANCE_RGCN,
        model=checkpoint_path,
        device=device,
        encoder=encoder,
        execution_requests=execution_requests,
    )
    from graph_memory.models.graph_retriever.provenance import (
        provenance_control_summary,
    )

    control = provenance_control_summary(
        [
            request.graph
            for request in execution_requests
            if isinstance(request, ExecutionProvenanceRankingRequest)
        ],
        model_config=checkpoint.model_config,
    )
    built[1]["control"] = cast(dict[str, RecursiveJsonValue], control)
    return built


def _build_evidence_rgcn(
    checkpoint_path: Path,
    method_id: RetrievalMethodId,
    text_requests: list[TextRankingRequest],
    graphs: list[EvidenceGraph],
    *,
    dense_encoder: SentenceEncoder | None,
    text_embedding_provider: TextEmbeddingProvider | None,
    seed_signal_provider: SeedSignalProvider | None,
    device: str,
) -> tuple[RetrievalMethod, RetrievalProvenance, list[RankingMethodRequest]]:
    from graph_memory.models.graph_retriever.inference import (
        CheckpointGraphRetrieverLoader,
    )

    text_provider, seed_provider, checkpoint = _evidence_rgcn_providers(
        checkpoint_path,
        method_id,
        dense_encoder=dense_encoder,
        text_embedding_provider=text_embedding_provider,
        seed_signal_provider=seed_signal_provider,
        device=device,
    )
    retrieval_method = CheckpointGraphRetrieverLoader().load(
        checkpoint_path,
        text_embedding_provider=text_provider,
        seed_signal_provider=seed_provider,
        device=device,
        expected_method=method_id,
    )
    return _built(
        retrieval_method,
        method=method_id,
        model=checkpoint_path,
        device=device,
        encoder=_checkpoint_encoder(checkpoint),
        execution_requests=_evidence_requests(
            text_requests,
            GraphIndex.from_graphs(graphs),
            _initial_scores_from_seed_signal_provider(seed_provider),
        ),
    )


def _evidence_rgcn_providers(
    checkpoint_path: Path,
    method_id: RetrievalMethodId,
    *,
    dense_encoder: SentenceEncoder | None,
    text_embedding_provider: TextEmbeddingProvider | None,
    seed_signal_provider: SeedSignalProvider | None,
    device: str,
):
    from graph_memory.models.graph_retriever.checkpoint import load_rgcn_checkpoint
    from graph_memory.models.graph_retriever.text_embeddings import (
        DenseGraphFeatureProvider,
    )
    from graph_memory.retrieval.signals import RetrieverSeedSignalProvider

    checkpoint = load_rgcn_checkpoint(
        checkpoint_path,
        expected_method=method_id,
        map_location=device,
    )
    if text_embedding_provider is not None and seed_signal_provider is not None:
        return text_embedding_provider, seed_signal_provider, checkpoint
    if text_embedding_provider is None and seed_signal_provider is None:
        joint_provider = DenseGraphFeatureProvider(
            model_name=checkpoint.model_config.encoder_model,
            query_prefix=checkpoint.model_config.query_prefix,
            passage_prefix=checkpoint.model_config.passage_prefix,
            device=device,
            encoder=dense_encoder,
        )
        return joint_provider, joint_provider, checkpoint
    if text_embedding_provider is None:
        text_embedding_provider = DenseGraphFeatureProvider(
            model_name=checkpoint.model_config.encoder_model,
            query_prefix=checkpoint.model_config.query_prefix,
            passage_prefix=checkpoint.model_config.passage_prefix,
            device=device,
            encoder=dense_encoder,
        )
    if seed_signal_provider is None:
        encoder = getattr(text_embedding_provider, "encoder", dense_encoder)
        seed_signal_provider = RetrieverSeedSignalProvider(
            DenseTaskRetriever(
                model_name=checkpoint.model_config.encoder_model,
                query_prefix=checkpoint.model_config.query_prefix,
                passage_prefix=checkpoint.model_config.passage_prefix,
                encoder=cast(SentenceEncoder | None, encoder),
                device=device,
            )
        )
    return text_embedding_provider, seed_signal_provider, checkpoint


def _encoder_settings(
    config: DenseEncoderConfig,
    model_name: str | None,
) -> DenseEncoderConfig:
    return DenseEncoderConfig(
        model_name=model_name or config.model_name,
        query_prefix=config.query_prefix,
        passage_prefix=config.passage_prefix,
        batch_size=config.batch_size,
    )


def _checkpoint_encoder(checkpoint: RgcnCheckpoint) -> DenseEncoderConfig:
    model_config = checkpoint.model_config
    return DenseEncoderConfig(
        model_name=model_config.encoder_model,
        query_prefix=model_config.query_prefix,
        passage_prefix=model_config.passage_prefix,
        batch_size=model_config.encoder_batch_size,
    )


def _resolve_encoder(
    settings: DenseEncoderConfig,
    encoder: SentenceEncoder | None,
    *,
    device: str,
) -> SentenceEncoder:
    if encoder is not None:
        return encoder
    try:
        return cast(
            SentenceEncoder,
            cast(object, load_sentence_transformer(settings.model_name, device=device)),
        )
    except RuntimeError as error:
        raise RuntimeError(
            "sentence-transformers is required for dense graph retrieval."
        ) from error


def _build_dense_ranker(
    settings: DenseEncoderConfig,
    encoder: SentenceEncoder | None,
    *,
    device: str,
) -> DenseTaskRetriever:
    return DenseTaskRetriever(
        config=DenseConfig(
            model_name=settings.model_name,
            query_prefix=settings.query_prefix,
            passage_prefix=settings.passage_prefix,
            batch_size=settings.batch_size,
            device=device,
        ),
        encoder=encoder or _resolve_encoder(settings, None, device=device),
        device=device,
    )


def _provenance_requests(
    text_requests: list[TextRankingRequest],
    graphs: list[ProvenanceGraph],
    graph_ids_by_task_id: Mapping[str, str],
    *,
    method: str,
) -> list[RankingMethodRequest]:
    graph_by_id = {graph.graph_id: graph for graph in graphs}
    if len(graph_by_id) != len(graphs):
        raise ValueError(f"{method} graph IDs must be unique")
    request_ids = {request.task_id for request in text_requests}
    if set(graph_ids_by_task_id) != request_ids:
        raise ValueError(f"{method} task-to-graph bindings must cover requests")
    result: list[RankingMethodRequest] = []
    for request in text_requests:
        graph_id = graph_ids_by_task_id[request.task_id]
        try:
            graph = graph_by_id[graph_id]
        except KeyError as error:
            raise ValueError(
                f"{method} task={request.task_id} references missing graph={graph_id}"
            ) from error
        result.append(
            ExecutionProvenanceRankingRequest(
                task_id=request.task_id,
                query_text=request.query_text,
                candidates=request.candidates,
                graph=graph,
            )
        )
    return result


def _built(
    retrieval_method: RetrievalMethod,
    *,
    method: RetrievalMethodId,
    execution_requests: list[RankingMethodRequest],
    model: Path | None = None,
    device: str | None = None,
    encoder: DenseEncoderConfig | None = None,
) -> tuple[RetrievalMethod, RetrievalProvenance, list[RankingMethodRequest]]:
    return (
        retrieval_method,
        RetrievalProvenance(
            method=method.value,
            model=None if model is None else model.as_posix(),
            device=device,
            encoder=None if encoder is None else encoder.model_dump(mode="json"),
        ),
        execution_requests,
    )


def _evidence_requests(
    text_requests: list[TextRankingRequest],
    graph_index: GraphIndex,
    initial_scores_for_request: Callable[[TextRankingRequest], dict[str, float]],
) -> list[RankingMethodRequest]:
    return [
        EvidenceGraphRankingRequest(
            task_id=request.task_id,
            query_text=request.query_text,
            candidates=request.candidates,
            graph=graph_index.get_required(request.task_id),
            initial_scores=initial_scores_for_request(request),
        )
        for request in text_requests
    ]


def _initial_scores_from_seed_signal_provider(
    seed_signal_provider: SeedSignalProvider,
) -> Callable[[TextRankingRequest], dict[str, float]]:
    def initial_scores(request: TextRankingRequest) -> dict[str, float]:
        return {
            signal.node_id: signal.score
            for signal in seed_signal_provider.score_task(request)
        }

    return initial_scores


def _required_checkpoint(checkpoint: Path | None, method: str) -> Path:
    if checkpoint is None:
        raise ValueError(f"trainable retrieval method={method} requires a checkpoint")
    return checkpoint


__all__ = [
    "RetrievalProvenance",
    "build_retrieval",
]

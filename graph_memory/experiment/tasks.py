from __future__ import annotations

from pathlib import Path

from prefect import task
from prefect.cache_policies import TASK_SOURCE
from prefect.logging import get_run_logger
from prefect.settings import (
    PREFECT_LOCAL_STORAGE_PATH,
    PREFECT_TASKS_REFRESH_CACHE,
    temporary_settings,
)

from graph_memory.experiment.artifacts import (
    DatasetArtifactRef,
    DirectorySourceRef,
    EvaluationArtifactRef,
    EvidenceGraphArtifactRef,
    FileSourceRef,
    FrozenEmbeddingsArtifactRef,
    ModelArtifactRef,
    PredictionsArtifactRef,
    ProcessedAssetStore,
    RevisionSourceRef,
    TrainingPairsArtifactRef,
    identify_external_source,
    identify_immutable_revision,
)
from graph_memory.experiment.cache import ScientificInputs
from graph_memory.experiment.config import (
    DatasetName,
    DenseEncoderConfig,
    DenseFinetuneMethodConfig,
    GraphBuildConfig,
    MethodConfig,
    PairBuildConfig,
    PrepareSplitConfig,
    ProvenanceRgcnMethodConfig,
    RgcnStageConfig,
    SplitName,
)
from graph_memory.stages.encodings import materialize_frozen_rgcn_embeddings
from graph_memory.stages.evaluate import materialize_evaluation
from graph_memory.stages.graphs import materialize_evidence_graphs
from graph_memory.stages.models import (
    materialize_dense_finetune_model,
    materialize_evidence_rgcn_model,
    materialize_provenance_rgcn_model,
)
from graph_memory.stages.pairs import materialize_training_pairs
from graph_memory.stages.prepare import materialize_prepared_split
from graph_memory.stages.retrieve import materialize_rankings


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_ROOT = REPOSITORY_ROOT / "data" / "processed"
SCIENTIFIC_CACHE_POLICY = ScientificInputs() + TASK_SOURCE
SCIENTIFIC_RESULT_STORAGE = PROCESSED_ROOT / "prefect" / "results"


def prefect_storage_settings(*, refresh_cache: bool):
    SCIENTIFIC_RESULT_STORAGE.mkdir(parents=True, exist_ok=True)
    return temporary_settings(
        updates={
            PREFECT_LOCAL_STORAGE_PATH: SCIENTIFIC_RESULT_STORAGE,
            PREFECT_TASKS_REFRESH_CACHE: refresh_cache,
        }
    )


def _processed_store() -> ProcessedAssetStore:
    return ProcessedAssetStore(PROCESSED_ROOT)


@task(
    name="prepare-split",
    persist_result=True,
    cache_policy=SCIENTIFIC_CACHE_POLICY,
)
def prepare_split_task(
    source: FileSourceRef,
    config: PrepareSplitConfig,
    trajectory_source: FileSourceRef | DirectorySourceRef | None = None,
    authoring_metadata_source: FileSourceRef | None = None,
    implementation_version: str = "prepare-v10-fixed-dev-tail",
) -> DatasetArtifactRef:
    get_run_logger().info(
        "prepare split | dataset=%s split=%s count=%s",
        config.dataset,
        config.split,
        config.count,
    )
    return materialize_prepared_split(
        _processed_store(),
        dataset=config.dataset,
        split=config.split,
        source=source,
        trajectory_source=trajectory_source,
        authoring_metadata_source=authoring_metadata_source,
        count=config.count,
        seed=config.seed,
        offset=config.offset,
        strict_invalid_examples=config.strict_invalid_examples,
        source_revision=config.source_revision,
        trajectory_splits=config.trajectory_splits,
        chunking=config.chunking,
        implementation_version=implementation_version,
    )


@task(
    name="build-evidence-graphs",
    persist_result=True,
    cache_policy=SCIENTIFIC_CACHE_POLICY,
)
def build_evidence_graphs_task(
    prepared: DatasetArtifactRef,
    dataset: DatasetName,
    split: SplitName,
    graph: GraphBuildConfig,
    implementation_version: str = "evidence-graphs-v2-isetrace",
) -> EvidenceGraphArtifactRef:
    get_run_logger().info("build evidence graphs | dataset=%s split=%s", dataset, split)
    return materialize_evidence_graphs(
        _processed_store(),
        dataset=dataset,
        split=split,
        prepared=prepared,
        config=graph,
        implementation_version=implementation_version,
    )


@task(
    name="build-training-pairs",
    persist_result=True,
    cache_policy=SCIENTIFIC_CACHE_POLICY,
)
def build_training_pairs_task(
    prepared: DatasetArtifactRef,
    evidence_graphs: EvidenceGraphArtifactRef | None,
    dataset: DatasetName,
    config: PairBuildConfig,
    encoder_source: FileSourceRef | DirectorySourceRef | RevisionSourceRef,
    implementation_version: str = "training-pairs-v2-dense-candidate-view",
) -> TrainingPairsArtifactRef:
    get_run_logger().info("build training pairs | dataset=%s", dataset)
    return materialize_training_pairs(
        _processed_store(),
        dataset=dataset,
        prepared=prepared,
        evidence_graphs=evidence_graphs,
        config=config,
        encoder_source=encoder_source,
        implementation_version=implementation_version,
    )


@task(
    name="encode-frozen-rgcn-embeddings",
    persist_result=True,
    cache_policy=SCIENTIFIC_CACHE_POLICY,
)
def encode_frozen_rgcn_embeddings_task(
    train_prepared: DatasetArtifactRef,
    dev_prepared: DatasetArtifactRef,
    train_graphs: EvidenceGraphArtifactRef | None,
    dev_graphs: EvidenceGraphArtifactRef | None,
    seed_model: ModelArtifactRef | None,
    dataset: DatasetName,
    encoder: DenseEncoderConfig,
    encoder_source: FileSourceRef | DirectorySourceRef | RevisionSourceRef,
    enable_gpupool: bool,
    device: str,
    chunk_size: int,
    implementation_version: str = "frozen-rgcn-embeddings-v1",
) -> FrozenEmbeddingsArtifactRef:
    get_run_logger().info(
        "encode frozen R-GCN embeddings | dataset=%s device=%s gpupool=%s",
        dataset,
        device,
        enable_gpupool,
    )
    return materialize_frozen_rgcn_embeddings(
        _processed_store(),
        dataset=dataset,
        encoder=encoder,
        encoder_source=encoder_source,
        train_prepared=train_prepared,
        dev_prepared=dev_prepared,
        train_graphs=train_graphs,
        dev_graphs=dev_graphs,
        seed_model=seed_model,
        enable_gpupool=enable_gpupool,
        device=device,
        chunk_size=chunk_size,
        implementation_version=implementation_version,
    )


@task(
    name="train-dense-ft",
    persist_result=True,
    cache_policy=SCIENTIFIC_CACHE_POLICY,
)
def train_dense_ft_task(
    train_prepared: DatasetArtifactRef,
    train_pairs: TrainingPairsArtifactRef,
    dev_prepared: DatasetArtifactRef,
    dataset: DatasetName,
    config: DenseFinetuneMethodConfig,
    encoder_source: FileSourceRef | DirectorySourceRef | RevisionSourceRef,
    implementation_version: str = "dense-ft-train-v2-candidate-view",
) -> ModelArtifactRef:
    get_run_logger().info(
        "train dense-ft | dataset=%s epochs=%s", dataset, config.train.trainer.epochs
    )
    return materialize_dense_finetune_model(
        _processed_store(),
        dataset=dataset,
        config=config,
        train_prepared=train_prepared,
        train_pairs=train_pairs,
        dev_prepared=dev_prepared,
        encoder_source=encoder_source,
        implementation_version=implementation_version,
    )


@task(
    name="train-evidence-rgcn",
    persist_result=True,
    cache_policy=SCIENTIFIC_CACHE_POLICY,
)
def train_evidence_rgcn_task(
    train_prepared: DatasetArtifactRef,
    train_graphs: EvidenceGraphArtifactRef,
    train_pairs: TrainingPairsArtifactRef,
    dev_prepared: DatasetArtifactRef,
    dev_graphs: EvidenceGraphArtifactRef,
    seed_model: ModelArtifactRef | None,
    dataset: DatasetName,
    method: str,
    variant: str,
    config: RgcnStageConfig,
    encoder_source: FileSourceRef | DirectorySourceRef | RevisionSourceRef,
    frozen_embeddings: FrozenEmbeddingsArtifactRef,
    implementation_version: str = "evidence-rgcn-train-v2-preencoded",
) -> ModelArtifactRef:
    get_run_logger().info(
        "train evidence-rgcn | dataset=%s epochs=%s",
        dataset,
        config.train.trainer.epochs,
    )
    return materialize_evidence_rgcn_model(
        _processed_store(),
        dataset=dataset,
        method=method,
        variant=variant,
        config=config,
        train_prepared=train_prepared,
        train_graphs=train_graphs,
        train_pairs=train_pairs,
        dev_prepared=dev_prepared,
        dev_graphs=dev_graphs,
        encoder_source=encoder_source,
        seed_model=seed_model,
        frozen_embeddings=frozen_embeddings,
        implementation_version=implementation_version,
    )


@task(
    name="train-provenance-rgcn",
    persist_result=True,
    cache_policy=SCIENTIFIC_CACHE_POLICY,
)
def train_provenance_rgcn_task(
    train_prepared: DatasetArtifactRef,
    train_pairs: TrainingPairsArtifactRef,
    dev_prepared: DatasetArtifactRef,
    config: ProvenanceRgcnMethodConfig,
    encoder_source: FileSourceRef | DirectorySourceRef | RevisionSourceRef,
    frozen_embeddings: FrozenEmbeddingsArtifactRef,
    implementation_version: str = "provenance-rgcn-train-v1",
) -> ModelArtifactRef:
    get_run_logger().info(
        "train provenance-rgcn | epochs=%s", config.train.trainer.epochs
    )
    return materialize_provenance_rgcn_model(
        _processed_store(),
        config=config,
        train_prepared=train_prepared,
        train_pairs=train_pairs,
        dev_prepared=dev_prepared,
        encoder_source=encoder_source,
        frozen_embeddings=frozen_embeddings,
        implementation_version=implementation_version,
    )


@task(
    name="generate-rankings",
    persist_result=True,
    cache_policy=SCIENTIFIC_CACHE_POLICY,
)
def generate_rankings_task(
    prepared: DatasetArtifactRef,
    evidence_graphs: EvidenceGraphArtifactRef | None,
    model: ModelArtifactRef | None,
    dataset: DatasetName,
    method: MethodConfig,
    top_k: int,
    encoder_source: FileSourceRef | DirectorySourceRef | RevisionSourceRef | None,
    device: str,
    implementation_version: str = "ranking-v2-device-aware",
) -> PredictionsArtifactRef:
    get_run_logger().info(
        "generate rankings | dataset=%s method=%s top_k=%s",
        dataset,
        method.method,
        top_k,
    )
    return materialize_rankings(
        _processed_store(),
        dataset=dataset,
        method=method,
        top_k=top_k,
        prepared=prepared,
        evidence_graphs=evidence_graphs,
        model=model,
        encoder_source=encoder_source,
        device=device,
        implementation_version=implementation_version,
    )


@task(
    name="evaluate-rankings",
    persist_result=True,
    cache_policy=SCIENTIFIC_CACHE_POLICY,
)
def evaluate_rankings_task(
    predictions: PredictionsArtifactRef,
    prepared: DatasetArtifactRef,
    evidence_graphs: EvidenceGraphArtifactRef | None,
    dataset: DatasetName,
    top_k: int,
    failure_case_limit: int,
    implementation_version: str = "evaluation-v5-isetrace-cluster-metadata",
) -> EvaluationArtifactRef:
    get_run_logger().info("evaluate rankings | dataset=%s top_k=%s", dataset, top_k)
    return materialize_evaluation(
        _processed_store(),
        dataset=dataset,
        top_k=top_k,
        failure_case_limit=failure_case_limit,
        predictions=predictions,
        prepared=prepared,
        evidence_graphs=evidence_graphs,
        implementation_version=implementation_version,
    )


def resolve_encoder_source(
    encoder: DenseEncoderConfig,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> DirectorySourceRef | FileSourceRef | RevisionSourceRef:
    configured = Path(encoder.model_name)
    local = configured if configured.is_absolute() else repository_root / configured
    if local.exists():
        return identify_external_source(local, repository_root=repository_root)
    if "@" in encoder.model_name:
        uri, revision = encoder.model_name.rsplit("@", maxsplit=1)
        return identify_immutable_revision(uri, revision)
    raise ValueError(
        "encoder model must resolve to a local content-addressed path or an immutable "
        f"name@revision: {encoder.model_name!r}"
    )


__all__ = [
    "PROCESSED_ROOT",
    "SCIENTIFIC_CACHE_POLICY",
    "SCIENTIFIC_RESULT_STORAGE",
    "build_evidence_graphs_task",
    "build_training_pairs_task",
    "encode_frozen_rgcn_embeddings_task",
    "evaluate_rankings_task",
    "generate_rankings_task",
    "prepare_split_task",
    "prefect_storage_settings",
    "resolve_encoder_source",
    "train_dense_ft_task",
    "train_evidence_rgcn_task",
    "train_provenance_rgcn_task",
]

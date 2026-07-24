from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import cast

from prefect import task
from prefect.cache_policies import INPUTS, TASK_SOURCE
from prefect.logging import get_run_logger
from prefect.settings import (
    PREFECT_LOCAL_STORAGE_PATH,
    PREFECT_TASKS_REFRESH_CACHE,
    temporary_settings,
)

from graph_memory.contracts.graphs import EvidenceGraph
from graph_memory.experiment.artifacts import (
    DatasetArtifactRef,
    DirectorySourceRef,
    EvidenceGraphArtifactRef,
    FileSourceRef,
    ModelArtifactRef,
    PredictionsArtifactRef,
    ProcessedAssetStore,
    RevisionSourceRef,
    TrainingPairsArtifactRef,
    artifact_payload_path,
    identify_external_source,
    identify_immutable_revision,
)
from graph_memory.datasets.twowiki_provenance import (
    TWOWIKI_PROVENANCE_SCHEMA_VERSION,
)
from graph_memory.experiment.config import (
    DatasetName,
    DenseEncoderConfig,
    DenseFinetuneStageConfig,
    GraphBuildConfig,
    PairBuildConfig,
    PrepareSplitConfig,
    ProvenanceRgcnStageConfig,
    RankingMethodConfig,
    RgcnTrainStageConfig,
    SplitName,
    TwoWikiProvenanceTransformConfig,
)
from graph_memory.io import read_json
from graph_memory.stages.evaluate import materialize_evaluation
from graph_memory.stages.graphs import materialize_evidence_graphs
from graph_memory.stages.models import (
    materialize_dense_finetune_model,
    materialize_evidence_rgcn_model,
    materialize_provenance_rgcn_model,
)
from graph_memory.stages.pairs import materialize_training_pairs
from graph_memory.stages.prepare import materialize_prepared_split
from graph_memory.stages.results import (
    BenchmarkResult,
    EvaluationResult,
    EvidenceGraphResult,
    ModelResult,
    PreparedSplitResult,
    RankingResult,
    TrainingPairsResult,
)
from graph_memory.stages.retrieve import materialize_rankings, run_retrieve_stage
from graph_memory.stages.transform import (
    TwoWikiProvenanceTransformResult,
    materialize_transform_twowiki,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_ROOT = REPOSITORY_ROOT / "data" / "processed"
TWOWIKI_PROVENANCE_RAW_ROOT = (
    REPOSITORY_ROOT / "data" / "twowiki_provenance" / "raw"
)


SCIENTIFIC_CACHE_POLICY = INPUTS + TASK_SOURCE
SCIENTIFIC_RESULT_STORAGE = PROCESSED_ROOT / "prefect" / "results"


def prefect_storage_settings(*, refresh_cache: bool):
    SCIENTIFIC_RESULT_STORAGE.mkdir(parents=True, exist_ok=True)
    return temporary_settings(
        updates={
            PREFECT_LOCAL_STORAGE_PATH: SCIENTIFIC_RESULT_STORAGE,
            PREFECT_TASKS_REFRESH_CACHE: refresh_cache,
        }
    )


def processed_store() -> ProcessedAssetStore:
    return ProcessedAssetStore(PROCESSED_ROOT)


@task(
    name="transform-twowiki",
    persist_result=True,
    cache_policy=SCIENTIFIC_CACHE_POLICY,
)
def transform_twowiki_task(
    train_source: FileSourceRef,
    dev_source: FileSourceRef,
    config: TwoWikiProvenanceTransformConfig,
    encoder_source: FileSourceRef | DirectorySourceRef | RevisionSourceRef | None,
    split_seed: int,
    schema_version: int = TWOWIKI_PROVENANCE_SCHEMA_VERSION,
) -> TwoWikiProvenanceTransformResult:
    get_run_logger().info(
        "transform twowiki_provenance | edge_scorer=%s", config.edge_scorer
    )
    if encoder_source is None:
        encoder_digest = None
    elif isinstance(encoder_source, RevisionSourceRef):
        encoder_digest = f"{encoder_source.uri}@{encoder_source.revision}"
    else:
        encoder_digest = encoder_source.digest
    return materialize_transform_twowiki(
        train_source=train_source,
        dev_source=dev_source,
        config=config,
        schema_version=schema_version,
        output_root=TWOWIKI_PROVENANCE_RAW_ROOT,
        repository_root=REPOSITORY_ROOT,
        encoder_digest=encoder_digest,
        split_seed=split_seed,
    )


@task(
    name="prepare-split",
    persist_result=True,
    cache_policy=SCIENTIFIC_CACHE_POLICY,
)
def prepare_split_task(
    source: FileSourceRef,
    config: PrepareSplitConfig,
    implementation_version: str = "prepare-v1",
) -> PreparedSplitResult:
    get_run_logger().info(
        "prepare split | dataset=%s split=%s count=%s",
        config.dataset,
        config.split,
        config.count,
    )
    return materialize_prepared_split(
        processed_store(),
        dataset=config.dataset,
        split=config.split,
        source=source,
        count=config.count,
        seed=config.seed,
        offset=config.offset,
        strict_invalid_examples=config.strict_invalid_examples,
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
    implementation_version: str = "evidence-graphs-v1",
) -> EvidenceGraphResult:
    get_run_logger().info(
        "build evidence graphs | dataset=%s split=%s", dataset, split
    )
    return materialize_evidence_graphs(
        processed_store(),
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
    implementation_version: str = "training-pairs-v1",
) -> TrainingPairsResult:
    get_run_logger().info("build training pairs | dataset=%s", dataset)
    return materialize_training_pairs(
        processed_store(),
        dataset=dataset,
        prepared=prepared,
        evidence_graphs=evidence_graphs,
        config=config,
        encoder_source=encoder_source,
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
    config: DenseFinetuneStageConfig,
    encoder_source: FileSourceRef | DirectorySourceRef | RevisionSourceRef,
    implementation_version: str = "dense-ft-train-v1",
) -> ModelResult:
    get_run_logger().info(
        "train dense-ft | dataset=%s epochs=%s", dataset, config.train.trainer.epochs
    )
    return materialize_dense_finetune_model(
        processed_store(),
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
    config: RgcnTrainStageConfig,
    encoder_source: FileSourceRef | DirectorySourceRef | RevisionSourceRef,
    implementation_version: str = "evidence-rgcn-train-v1",
) -> ModelResult:
    get_run_logger().info(
        "train evidence-rgcn | dataset=%s epochs=%s", dataset, config.train.trainer.epochs
    )
    return materialize_evidence_rgcn_model(
        processed_store(),
        dataset=dataset,
        config=config,
        train_prepared=train_prepared,
        train_graphs=train_graphs,
        train_pairs=train_pairs,
        dev_prepared=dev_prepared,
        dev_graphs=dev_graphs,
        encoder_source=encoder_source,
        seed_model=seed_model,
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
    dataset: DatasetName,
    config: ProvenanceRgcnStageConfig,
    encoder_source: FileSourceRef | DirectorySourceRef | RevisionSourceRef,
    implementation_version: str = "provenance-rgcn-train-v1",
) -> ModelResult:
    get_run_logger().info(
        "train provenance-rgcn | dataset=%s epochs=%s", dataset, config.train.trainer.epochs
    )
    return materialize_provenance_rgcn_model(
        processed_store(),
        dataset=dataset,
        config=config,
        train_prepared=train_prepared,
        train_pairs=train_pairs,
        dev_prepared=dev_prepared,
        encoder_source=encoder_source,
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
    method: RankingMethodConfig,
    top_k: int,
    encoder_source: FileSourceRef | DirectorySourceRef | RevisionSourceRef | None,
    device: str,
    implementation_version: str = "ranking-v2-device-aware",
) -> RankingResult:
    get_run_logger().info(
        "generate rankings | dataset=%s method=%s top_k=%s",
        dataset,
        method.method,
        top_k,
    )
    return materialize_rankings(
        processed_store(),
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
    implementation_version: str = "evaluation-v1",
) -> EvaluationResult:
    get_run_logger().info("evaluate rankings | dataset=%s top_k=%s", dataset, top_k)
    return materialize_evaluation(
        processed_store(),
        dataset=dataset,
        top_k=top_k,
        failure_case_limit=failure_case_limit,
        predictions=predictions,
        prepared=prepared,
        evidence_graphs=evidence_graphs,
        implementation_version=implementation_version,
    )


@task(
    name="benchmark-retrieval",
    persist_result=False,
)
def benchmark_retrieval_task(
    prepared: DatasetArtifactRef,
    evidence_graphs: EvidenceGraphArtifactRef | None,
    model: ModelArtifactRef | None,
    dataset: DatasetName,
    method: RankingMethodConfig,
    top_k: int,
    encoder_source: FileSourceRef | DirectorySourceRef | RevisionSourceRef | None,
    device: str,
    warmup: int,
    repetitions: int,
) -> BenchmarkResult:
    get_run_logger().info(
        "benchmark retrieval | dataset=%s warmup=%s repetitions=%s",
        dataset,
        warmup,
        repetitions,
    )
    task_inputs = read_json(artifact_payload_path(prepared, "tasks"))
    graph_values = (
        cast(
            list[EvidenceGraph],
            read_json(
                artifact_payload_path(evidence_graphs, "graphs")
            ),
        )
        if evidence_graphs is not None
        else []
    )
    durations: list[float] = []
    for index in range(warmup + repetitions):
        started = time.perf_counter()
        _ = run_retrieve_stage(
            method,
            dataset=dataset,
            top_k=top_k,
            task_inputs=task_inputs,
            evidence_graphs=graph_values,
            model=model,
            encoder_source=encoder_source,
            device=device,
        )
        elapsed = time.perf_counter() - started
        if index >= warmup:
            durations.append(elapsed)
    if not durations:
        raise ValueError("benchmark requires at least one measured repetition")
    query_count = max(1, len(task_inputs))
    mean_seconds = statistics.fmean(durations)
    return BenchmarkResult(
        warmup=warmup,
        repetitions=repetitions,
        metrics={
            "benchmark.retrieval_seconds_mean": mean_seconds,
            "benchmark.retrieval_latency_ms_per_query": (
                mean_seconds * 1000.0 / query_count
            ),
        },
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
    "benchmark_retrieval_task",
    "build_evidence_graphs_task",
    "build_training_pairs_task",
    "evaluate_rankings_task",
    "generate_rankings_task",
    "prepare_split_task",
    "prefect_storage_settings",
    "processed_store",
    "resolve_encoder_source",
    "train_dense_ft_task",
    "train_evidence_rgcn_task",
    "train_provenance_rgcn_task",
    "transform_twowiki_task",
]

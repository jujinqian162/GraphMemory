from __future__ import annotations

from pathlib import Path

from prefect import flow
from prefect.runtime import flow_run

from graph_memory.experiment.artifacts import (
    ArtifactRef,
    DatasetArtifactRef,
    DirectorySourceRef,
    EvidenceGraphArtifactRef,
    FileSourceRef,
    ModelArtifactRef,
    TrainingPairsArtifactRef,
    identify_external_source,
)
from graph_memory.experiment.config import (
    Bm25MethodConfig,
    DatasetName,
    DenseFinetuneMethodConfig,
    DenseFtRgcnMethodConfig,
    DenseMethodConfig,
    GraphRAGMethodConfig,
    MethodConfig,
    PairBuildConfig,
    ProvenancePathMethodConfig,
    ProvenanceRgcnMethodConfig,
    PrepareSplitConfig,
    ResolvedExperimentConfig,
    RgcnMethodConfig,
    SplitName,
)
from graph_memory.experiment.inputs import ensure_inputs
from graph_memory.query_synthesis.provenance import authoring_metadata_path
from graph_memory.experiment.output import project_run_output
from graph_memory.experiment.results import FinalExperimentResult
from graph_memory.experiment.tasks import (
    build_evidence_graphs_task,
    build_training_pairs_task,
    encode_frozen_rgcn_embeddings_task,
    evaluate_rankings_task,
    generate_rankings_task,
    prefect_storage_settings,
    prepare_split_task,
    resolve_encoder_source,
    train_dense_ft_task,
    train_evidence_rgcn_task,
    train_provenance_rgcn_task,
)
from graph_memory.experiment.tracking import log_experiment_result


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_SPLIT_NAMES: tuple[SplitName, ...] = ("train", "dev", "test")


def _provide_dense_ft_seed(
    *,
    seed: DenseFinetuneMethodConfig,
    train: DatasetArtifactRef,
    dev: DatasetArtifactRef,
    train_graphs: EvidenceGraphArtifactRef | None,
    dataset: DatasetName,
    device: str,
) -> tuple[TrainingPairsArtifactRef, ModelArtifactRef]:
    seed_source = resolve_encoder_source(seed.encoder)
    seed_pairs = build_training_pairs_task(
        prepared=train,
        evidence_graphs=train_graphs,
        dataset=dataset,
        config=PairBuildConfig(
            method=seed.method,
            candidate_view=seed.variant,
            sampling=seed.pairs,
            encoder=seed.encoder,
            device=device,
        ),
        encoder_source=seed_source,
    )
    seed_model = train_dense_ft_task(
        train_prepared=train,
        train_pairs=seed_pairs,
        dev_prepared=dev,
        dataset=dataset,
        config=seed,
        encoder_source=seed_source,
    )
    return seed_pairs, seed_model


@flow(name="graph-memory-experiment", persist_result=False)
def run_experiment(
    config: ResolvedExperimentConfig,
    *,
    run_output: Path,
    overrides: tuple[str, ...] = (),
) -> FinalExperimentResult:
    method = config.method
    model: ModelArtifactRef | None = None
    dependency_models: tuple[ModelArtifactRef, ...] = ()
    ranking_graphs: EvidenceGraphArtifactRef | None = None
    evaluation_graphs: EvidenceGraphArtifactRef | None = None
    ranking_encoder = None
    assets: list[ArtifactRef] = []

    ensure_inputs(config)

    with prefect_storage_settings(refresh_cache=config.cache.refresh):
        split_sources = _resolve_split_sources(config)
        trajectory_source = _trajectory_source(config)
        authoring_metadata_source = _authoring_metadata_source(config)
        requires_training = isinstance(
            method,
            (
                DenseFinetuneMethodConfig,
                ProvenanceRgcnMethodConfig,
                RgcnMethodConfig,
                DenseFtRgcnMethodConfig,
            ),
        )
        prepared = {
            split: prepare_split_task(
                source=split_sources[split],
                config=_prepare_config(config, split),
                trajectory_source=trajectory_source,
                authoring_metadata_source=authoring_metadata_source,
            )
            for split in (("train", "dev", "test") if requires_training else ("test",))
        }
        test = prepared["test"]
        assets.extend(prepared.values())

        if isinstance(
            method,
            (
                Bm25MethodConfig,
                DenseMethodConfig,
                GraphRAGMethodConfig,
                ProvenancePathMethodConfig,
            ),
        ):
            if not isinstance(method, Bm25MethodConfig):
                ranking_encoder = resolve_encoder_source(method.encoder)

        elif isinstance(method, DenseFinetuneMethodConfig):
            train = prepared["train"]
            dev = prepared["dev"]
            encoder_source = resolve_encoder_source(method.encoder)
            effective_pairs = method.pairs
            train_graphs = None
            if effective_pairs.hard_graph_neighbor_per_positive > 0:
                train_graphs = build_evidence_graphs_task(
                    prepared=train,
                    dataset=config.dataset.name,
                    split="train",
                    graph=config.graph,
                )
                assets.append(train_graphs)

            pairs = build_training_pairs_task(
                prepared=train,
                evidence_graphs=(
                    None if train_graphs is None else train_graphs
                ),
                dataset=config.dataset.name,
                config=PairBuildConfig(
                    method=method.method,
                    candidate_view=method.variant,
                    sampling=effective_pairs,
                    encoder=method.encoder,
                    device=config.device,
                ),
                encoder_source=encoder_source,
            )
            model = train_dense_ft_task(
                train_prepared=train,
                train_pairs=pairs,
                dev_prepared=dev,
                dataset=config.dataset.name,
                config=method,
                encoder_source=encoder_source,
            )
            assets.extend((pairs, model))

        elif isinstance(method, ProvenanceRgcnMethodConfig):
            train = prepared["train"]
            dev = prepared["dev"]
            encoder_source = resolve_encoder_source(method.encoder)
            seed_model = None
            if method.seed is not None:
                seed_pairs, seed_model = _provide_dense_ft_seed(
                    seed=method.seed,
                    train=train,
                    dev=dev,
                    train_graphs=None,
                    dataset=config.dataset.name,
                    device=config.device,
                )
                dependency_models = (seed_model,)
                assets.extend((seed_pairs, seed_model))

            pairs = build_training_pairs_task(
                prepared=train,
                evidence_graphs=None,
                dataset=config.dataset.name,
                config=PairBuildConfig(
                    method=method.method,
                    sampling=method.pairs,
                    encoder=method.encoder,
                    device=config.device,
                ),
                encoder_source=encoder_source,
            )
            frozen_embeddings = encode_frozen_rgcn_embeddings_task(
                train_prepared=train,
                dev_prepared=dev,
                train_graphs=None,
                dev_graphs=None,
                seed_model=seed_model,
                dataset=config.dataset.name,
                encoder=method.encoder,
                encoder_source=encoder_source,
                enable_gpupool=config.encoding.enable_gpupool,
                device=config.device,
                chunk_size=config.encoding.chunk_size,
            )
            model = train_provenance_rgcn_task(
                train_prepared=train,
                train_pairs=pairs,
                dev_prepared=dev,
                config=method,
                encoder_source=encoder_source,
                seed_model=seed_model,
                frozen_embeddings=frozen_embeddings,
            )
            assets.extend(
                (
                    pairs,
                    frozen_embeddings,
                    model,
                )
            )

        elif isinstance(method, RgcnMethodConfig):
            train = prepared["train"]
            dev = prepared["dev"]
            train_graphs = build_evidence_graphs_task(
                prepared=train,
                dataset=config.dataset.name,
                split="train",
                graph=config.graph,
            )
            dev_graphs = build_evidence_graphs_task(
                prepared=dev,
                dataset=config.dataset.name,
                split="dev",
                graph=config.graph,
            )
            test_graphs = build_evidence_graphs_task(
                prepared=test,
                dataset=config.dataset.name,
                split="test",
                graph=config.graph,
            )
            encoder_source = resolve_encoder_source(method.encoder)
            pairs = build_training_pairs_task(
                prepared=train,
                evidence_graphs=train_graphs,
                dataset=config.dataset.name,
                config=PairBuildConfig(
                    method=method.method,
                    sampling=method.pairs,
                    encoder=method.encoder,
                    device=config.device,
                ),
                encoder_source=encoder_source,
            )
            frozen_embeddings = encode_frozen_rgcn_embeddings_task(
                train_prepared=train,
                dev_prepared=dev,
                train_graphs=train_graphs,
                dev_graphs=dev_graphs,
                seed_model=None,
                dataset=config.dataset.name,
                encoder=method.encoder,
                encoder_source=encoder_source,
                enable_gpupool=config.encoding.enable_gpupool,
                device=config.device,
                chunk_size=config.encoding.chunk_size,
            )
            model = train_evidence_rgcn_task(
                train_prepared=train,
                train_graphs=train_graphs,
                train_pairs=pairs,
                dev_prepared=dev,
                dev_graphs=dev_graphs,
                seed_model=None,
                dataset=config.dataset.name,
                method=method.method,
                variant=method.variant,
                config=method,
                encoder_source=encoder_source,
                frozen_embeddings=frozen_embeddings,
            )
            ranking_graphs = test_graphs
            assets.extend(
                (
                    train_graphs,
                    dev_graphs,
                    test_graphs,
                    pairs,
                    frozen_embeddings,
                    model,
                )
            )

        elif isinstance(method, DenseFtRgcnMethodConfig):
            train = prepared["train"]
            dev = prepared["dev"]
            train_graphs = build_evidence_graphs_task(
                prepared=train,
                dataset=config.dataset.name,
                split="train",
                graph=config.graph,
            )
            dev_graphs = build_evidence_graphs_task(
                prepared=dev,
                dataset=config.dataset.name,
                split="dev",
                graph=config.graph,
            )
            test_graphs = build_evidence_graphs_task(
                prepared=test,
                dataset=config.dataset.name,
                split="test",
                graph=config.graph,
            )

            seed_pairs, seed_model = _provide_dense_ft_seed(
                seed=method.seed,
                train=train,
                dev=dev,
                train_graphs=train_graphs,
                dataset=config.dataset.name,
                device=config.device,
            )

            rgcn = method.rgcn
            rgcn_source = resolve_encoder_source(rgcn.encoder)
            rgcn_pairs = build_training_pairs_task(
                prepared=train,
                evidence_graphs=train_graphs,
                dataset=config.dataset.name,
                config=PairBuildConfig(
                    method=method.method,
                    sampling=rgcn.pairs,
                    encoder=rgcn.encoder,
                    device=config.device,
                ),
                encoder_source=rgcn_source,
            )
            frozen_embeddings = encode_frozen_rgcn_embeddings_task(
                train_prepared=train,
                dev_prepared=dev,
                train_graphs=train_graphs,
                dev_graphs=dev_graphs,
                seed_model=seed_model,
                dataset=config.dataset.name,
                encoder=rgcn.encoder,
                encoder_source=rgcn_source,
                enable_gpupool=config.encoding.enable_gpupool,
                device=config.device,
                chunk_size=config.encoding.chunk_size,
            )
            model = train_evidence_rgcn_task(
                train_prepared=train,
                train_graphs=train_graphs,
                train_pairs=rgcn_pairs,
                dev_prepared=dev,
                dev_graphs=dev_graphs,
                seed_model=seed_model,
                dataset=config.dataset.name,
                method=method.method,
                variant=method.variant,
                config=method.rgcn,
                encoder_source=rgcn_source,
                frozen_embeddings=frozen_embeddings,
            )
            dependency_models = (seed_model,)
            ranking_graphs = test_graphs
            assets.extend(
                (
                    train_graphs,
                    dev_graphs,
                    test_graphs,
                    seed_pairs,
                    seed_model,
                    rgcn_pairs,
                    frozen_embeddings,
                    model,
                )
            )

        else:
            raise ValueError(f"unsupported final method={type(method).__name__}")

        rank_config: MethodConfig = method
        ranking = generate_rankings_task(
            prepared=test,
            evidence_graphs=ranking_graphs,
            model=None if model is None else model,
            dataset=config.dataset.name,
            method=rank_config,
            top_k=config.top_k,
            encoder_source=ranking_encoder,
            device=config.device,
            implementation_version="ranking-v11-rgcn-seed-residual",
        )
        evaluation = evaluate_rankings_task(
            predictions=ranking,
            prepared=test,
            evidence_graphs=evaluation_graphs or ranking_graphs,
            dataset=config.dataset.name,
            top_k=config.top_k,
            failure_case_limit=50,
        )

    assets.extend((ranking, evaluation))
    final = FinalExperimentResult(
        method=method.method,
        variant=config.variant,
        model=model,
        dependency_models=dependency_models,
        ranking=ranking,
        evaluation=evaluation,
        assets=_unique_assets(assets),
    )
    project_run_output(
        run_output,
        repository_root=REPOSITORY_ROOT,
        config=config,
        overrides=overrides,
        result=final,
    )
    log_experiment_result(
        config,
        final,
        run_output=run_output,
        prefect_flow_run_id=str(flow_run.id),
    )
    return final


def _resolve_split_sources(
    config: ResolvedExperimentConfig,
) -> dict[SplitName, FileSourceRef]:
    return {
        split: _direct_split_source(config, split)
        for split in _SPLIT_NAMES
        if split in config.dataset.splits
    }


def _direct_split_source(
    config: ResolvedExperimentConfig,
    split: SplitName,
) -> FileSourceRef:
    source = identify_external_source(
        config.dataset.splits[split].source,
        repository_root=REPOSITORY_ROOT,
    )
    if not isinstance(source, FileSourceRef):
        raise TypeError(f"raw split source must be a file: {source.uri}")
    return source


def _authoring_metadata_source(
    config: ResolvedExperimentConfig,
) -> FileSourceRef | None:
    query_source = config.dataset.natural_query_source
    if query_source is None:
        return None
    source = identify_external_source(
        authoring_metadata_path(query_source),
        repository_root=REPOSITORY_ROOT,
    )
    if not isinstance(source, FileSourceRef):
        raise TypeError(f"authoring metadata source must be a file: {source.uri}")
    return source


def _trajectory_source(
    config: ResolvedExperimentConfig,
) -> FileSourceRef | DirectorySourceRef | None:
    source_path = config.dataset.trajectory_source
    if source_path is None:
        return None
    source = identify_external_source(
        source_path,
        repository_root=REPOSITORY_ROOT,
    )
    return source


def _prepare_config(
    config: ResolvedExperimentConfig,
    split: SplitName,
) -> PrepareSplitConfig:
    split_config = config.dataset.splits[split]
    # ISETrace query selection is deterministic within the immutable trajectory
    # partition. Evidence datasets retain their existing seed policy.
    sampling_seed = (
        config.split_seed
        if config.dataset.name == "isetrace" or split == "test"
        else config.seed
    )
    return PrepareSplitConfig(
        dataset=config.dataset.name,
        split=split,
        count=split_config.count,
        offset=split_config.offset,
        seed=sampling_seed,
        strict_invalid_examples=config.dataset.strict_invalid_examples,
        source_revision=config.dataset.source_revision,
        trajectory_splits=(
            None
            if config.dataset.trajectories is None
            else config.dataset.trajectories.splits
        ),
        chunking=config.dataset.chunking,
    )


def _unique_assets(values: list[ArtifactRef]) -> tuple[ArtifactRef, ...]:
    return tuple({(value.kind, value.digest): value for value in values}.values())


__all__ = ["run_experiment"]

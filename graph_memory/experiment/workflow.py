from __future__ import annotations

from pathlib import Path

from prefect import flow
from prefect.runtime import flow_run

from graph_memory.experiment.artifacts import (
    ArtifactRef,
    DirectorySourceRef,
    EvidenceGraphArtifactRef,
    FileSourceRef,
    identify_external_source,
)
from graph_memory.experiment.config import (
    Bm25MethodConfig,
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
from graph_memory.experiment.output import project_run_output
from graph_memory.experiment.inputs import ensure_inputs
from graph_memory.experiment.results import FinalExperimentResult
from graph_memory.experiment.tasks import (
    benchmark_retrieval_task,
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
from graph_memory.stages.results import (
    BenchmarkResult,
    ModelResult,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_SPLIT_NAMES: tuple[SplitName, ...] = ("train", "dev", "test")


@flow(name="graph-memory-experiment", persist_result=False)
def run_experiment(
    config: ResolvedExperimentConfig,
    *,
    run_output: Path,
    overrides: tuple[str, ...] = (),
) -> FinalExperimentResult:
    method = config.method
    model: ModelResult | None = None
    dependency_models: tuple[ModelResult, ...] = ()
    ranking_graphs: EvidenceGraphArtifactRef | None = None
    evaluation_graphs: EvidenceGraphArtifactRef | None = None
    ranking_encoder = None
    assets: list[ArtifactRef] = []

    ensure_inputs(config)

    with prefect_storage_settings(refresh_cache=config.cache.refresh):
        split_sources = _resolve_split_sources(config)
        if isinstance(
            method,
            (
                Bm25MethodConfig,
                DenseMethodConfig,
                GraphRAGMethodConfig,
                ProvenancePathMethodConfig,
            ),
        ):
            trajectory_source = _trajectory_source(config)
            test = prepare_split_task(
                source=split_sources["test"],
                config=_prepare_config(config, "test"),
                trajectory_source=trajectory_source,
            )
            assets.append(test.artifact)
            if not isinstance(method, Bm25MethodConfig):
                ranking_encoder = resolve_encoder_source(method.encoder)

        elif isinstance(method, DenseFinetuneMethodConfig):
            trajectory_source = _trajectory_source(config)
            train = prepare_split_task(
                source=split_sources["train"],
                config=_prepare_config(config, "train"),
                trajectory_source=trajectory_source,
            )
            dev = prepare_split_task(
                source=split_sources["dev"],
                config=_prepare_config(config, "dev"),
                trajectory_source=trajectory_source,
            )
            test = prepare_split_task(
                source=split_sources["test"],
                config=_prepare_config(config, "test"),
                trajectory_source=trajectory_source,
            )
            assets.extend((train.artifact, dev.artifact, test.artifact))

            encoder_source = resolve_encoder_source(method.encoder)
            effective_pairs = method.pairs
            train_graphs = None
            if effective_pairs.hard_graph_neighbor_per_positive > 0:
                train_graphs = build_evidence_graphs_task(
                    prepared=train.artifact,
                    dataset=config.dataset.name,
                    split=train.split,
                    graph=config.graph,
                )
                assets.append(train_graphs.artifact)

            pairs = build_training_pairs_task(
                prepared=train.artifact,
                evidence_graphs=(
                    None if train_graphs is None else train_graphs.artifact
                ),
                dataset=config.dataset.name,
                config=PairBuildConfig(
                    method=method.method,
                    sampling=effective_pairs,
                    encoder=method.encoder,
                    device=config.device,
                ),
                encoder_source=encoder_source,
            )
            model = train_dense_ft_task(
                train_prepared=train.artifact,
                train_pairs=pairs.artifact,
                dev_prepared=dev.artifact,
                dataset=config.dataset.name,
                config=method,
                encoder_source=encoder_source,
            )
            assets.extend((pairs.artifact, model.artifact))

        elif isinstance(method, ProvenanceRgcnMethodConfig):
            trajectory_source = _trajectory_source(config)
            train = prepare_split_task(
                source=split_sources["train"],
                config=_prepare_config(config, "train"),
                trajectory_source=trajectory_source,
            )
            dev = prepare_split_task(
                source=split_sources["dev"],
                config=_prepare_config(config, "dev"),
                trajectory_source=trajectory_source,
            )
            test = prepare_split_task(
                source=split_sources["test"],
                config=_prepare_config(config, "test"),
                trajectory_source=trajectory_source,
            )
            encoder_source = resolve_encoder_source(method.encoder)
            pairs = build_training_pairs_task(
                prepared=train.artifact,
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
                train_prepared=train.artifact,
                dev_prepared=dev.artifact,
                train_graphs=None,
                dev_graphs=None,
                seed_model=None,
                dataset=config.dataset.name,
                encoder=method.encoder,
                encoder_source=encoder_source,
                enable_gpupool=config.encoding.enable_gpupool,
                device=config.device,
                chunk_size=config.encoding.chunk_size,
            )
            model = train_provenance_rgcn_task(
                train_prepared=train.artifact,
                train_pairs=pairs.artifact,
                dev_prepared=dev.artifact,
                config=method,
                encoder_source=encoder_source,
                frozen_embeddings=frozen_embeddings.artifact,
            )
            assets.extend(
                (
                    train.artifact,
                    dev.artifact,
                    test.artifact,
                    pairs.artifact,
                    frozen_embeddings.artifact,
                    model.artifact,
                )
            )

        elif isinstance(method, RgcnMethodConfig):
            train = prepare_split_task(
                source=split_sources["train"],
                config=_prepare_config(config, "train"),
            )
            dev = prepare_split_task(
                source=split_sources["dev"],
                config=_prepare_config(config, "dev"),
            )
            test = prepare_split_task(
                source=split_sources["test"],
                config=_prepare_config(config, "test"),
            )
            train_graphs = build_evidence_graphs_task(
                prepared=train.artifact,
                dataset=config.dataset.name,
                split=train.split,
                graph=config.graph,
            )
            dev_graphs = build_evidence_graphs_task(
                prepared=dev.artifact,
                dataset=config.dataset.name,
                split=dev.split,
                graph=config.graph,
            )
            test_graphs = build_evidence_graphs_task(
                prepared=test.artifact,
                dataset=config.dataset.name,
                split=test.split,
                graph=config.graph,
            )
            encoder_source = resolve_encoder_source(method.encoder)
            pairs = build_training_pairs_task(
                prepared=train.artifact,
                evidence_graphs=train_graphs.artifact,
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
                train_prepared=train.artifact,
                dev_prepared=dev.artifact,
                train_graphs=train_graphs.artifact,
                dev_graphs=dev_graphs.artifact,
                seed_model=None,
                dataset=config.dataset.name,
                encoder=method.encoder,
                encoder_source=encoder_source,
                enable_gpupool=config.encoding.enable_gpupool,
                device=config.device,
                chunk_size=config.encoding.chunk_size,
            )
            model = train_evidence_rgcn_task(
                train_prepared=train.artifact,
                train_graphs=train_graphs.artifact,
                train_pairs=pairs.artifact,
                dev_prepared=dev.artifact,
                dev_graphs=dev_graphs.artifact,
                seed_model=None,
                dataset=config.dataset.name,
                method=method.method,
                variant=method.variant,
                config=method,
                encoder_source=encoder_source,
                frozen_embeddings=frozen_embeddings.artifact,
            )
            ranking_graphs = test_graphs.artifact
            assets.extend(
                (
                    train.artifact,
                    dev.artifact,
                    test.artifact,
                    train_graphs.artifact,
                    dev_graphs.artifact,
                    test_graphs.artifact,
                    pairs.artifact,
                    frozen_embeddings.artifact,
                    model.artifact,
                )
            )

        elif isinstance(method, DenseFtRgcnMethodConfig):
            train = prepare_split_task(
                source=split_sources["train"],
                config=_prepare_config(config, "train"),
            )
            dev = prepare_split_task(
                source=split_sources["dev"],
                config=_prepare_config(config, "dev"),
            )
            test = prepare_split_task(
                source=split_sources["test"],
                config=_prepare_config(config, "test"),
            )
            train_graphs = build_evidence_graphs_task(
                prepared=train.artifact,
                dataset=config.dataset.name,
                split=train.split,
                graph=config.graph,
            )
            dev_graphs = build_evidence_graphs_task(
                prepared=dev.artifact,
                dataset=config.dataset.name,
                split=dev.split,
                graph=config.graph,
            )
            test_graphs = build_evidence_graphs_task(
                prepared=test.artifact,
                dataset=config.dataset.name,
                split=test.split,
                graph=config.graph,
            )

            seed_source = resolve_encoder_source(method.seed.encoder)
            seed_pairs = build_training_pairs_task(
                prepared=train.artifact,
                evidence_graphs=train_graphs.artifact,
                dataset=config.dataset.name,
                config=PairBuildConfig(
                    method=method.seed.method,
                    sampling=method.seed.pairs,
                    encoder=method.seed.encoder,
                    device=config.device,
                ),
                encoder_source=seed_source,
            )
            seed_model = train_dense_ft_task(
                train_prepared=train.artifact,
                train_pairs=seed_pairs.artifact,
                dev_prepared=dev.artifact,
                dataset=config.dataset.name,
                config=method.seed,
                encoder_source=seed_source,
            )

            rgcn = method.rgcn
            rgcn_source = resolve_encoder_source(rgcn.encoder)
            rgcn_pairs = build_training_pairs_task(
                prepared=train.artifact,
                evidence_graphs=train_graphs.artifact,
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
                train_prepared=train.artifact,
                dev_prepared=dev.artifact,
                train_graphs=train_graphs.artifact,
                dev_graphs=dev_graphs.artifact,
                seed_model=seed_model.artifact,
                dataset=config.dataset.name,
                encoder=rgcn.encoder,
                encoder_source=rgcn_source,
                enable_gpupool=config.encoding.enable_gpupool,
                device=config.device,
                chunk_size=config.encoding.chunk_size,
            )
            model = train_evidence_rgcn_task(
                train_prepared=train.artifact,
                train_graphs=train_graphs.artifact,
                train_pairs=rgcn_pairs.artifact,
                dev_prepared=dev.artifact,
                dev_graphs=dev_graphs.artifact,
                seed_model=seed_model.artifact,
                dataset=config.dataset.name,
                method=method.method,
                variant=method.variant,
                config=method.rgcn,
                encoder_source=rgcn_source,
                frozen_embeddings=frozen_embeddings.artifact,
            )
            dependency_models = (seed_model,)
            ranking_graphs = test_graphs.artifact
            assets.extend(
                (
                    train.artifact,
                    dev.artifact,
                    test.artifact,
                    train_graphs.artifact,
                    dev_graphs.artifact,
                    test_graphs.artifact,
                    seed_pairs.artifact,
                    seed_model.artifact,
                    rgcn_pairs.artifact,
                    frozen_embeddings.artifact,
                    model.artifact,
                )
            )

        else:
            raise ValueError(f"unsupported final method={type(method).__name__}")

        rank_config: MethodConfig = method
        ranking = generate_rankings_task(
            prepared=test.artifact,
            evidence_graphs=ranking_graphs,
            model=None if model is None else model.artifact,
            dataset=config.dataset.name,
            method=rank_config,
            top_k=config.top_k,
            encoder_source=ranking_encoder,
            device=config.device,
            implementation_version="ranking-v9-fast-graphrag-ppr",
        )
        evaluation = evaluate_rankings_task(
            predictions=ranking.artifact,
            prepared=test.artifact,
            evidence_graphs=evaluation_graphs or ranking_graphs,
            dataset=config.dataset.name,
            top_k=config.top_k,
            failure_case_limit=config.evaluation.failure_case_limit,
        )
        benchmark: BenchmarkResult | None = None
        if config.benchmark.enabled:
            benchmark = benchmark_retrieval_task(
                prepared=test.artifact,
                evidence_graphs=ranking_graphs,
                model=None if model is None else model.artifact,
                dataset=config.dataset.name,
                method=rank_config,
                top_k=config.top_k,
                encoder_source=ranking_encoder,
                device=config.device,
                warmup=config.benchmark.warmup,
                repetitions=config.benchmark.repetitions,
            )

    assets.extend((ranking.artifact, evaluation.artifact))
    final = FinalExperimentResult(
        method=method.method,
        variant=config.variant,
        model=model,
        dependency_models=dependency_models,
        ranking=ranking,
        evaluation=evaluation,
        benchmark=benchmark,
        assets=_unique_assets(assets),
    )
    project_run_output(
        run_output,
        repository_root=REPOSITORY_ROOT,
        config=config,
        overrides=overrides,
        result=final,
    )
    completed = final.model_copy(update={"run_output": run_output.resolve().as_posix()})
    log_experiment_result(
        config,
        completed,
        run_output=run_output,
        prefect_flow_run_id=str(flow_run.id),
    )
    return completed


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
        count=None if config.dataset.name == "isetrace" else split_config.count,
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

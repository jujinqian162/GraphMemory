from __future__ import annotations

from pathlib import Path

from prefect import flow
from prefect.runtime import flow_run

from graph_memory.experiment.artifacts import (
    ArtifactRef,
    EvidenceGraphArtifactRef,
    FileSourceRef,
    identify_external_source,
)
from graph_memory.experiment.config import (
    Bm25MethodConfig,
    DenseEncoderConfig,
    DenseFinetuneMethodConfig,
    DenseFtRgcnMethodConfig,
    DenseMethodConfig,
    ExecutionProvenanceMethodConfig,
    ExecutionProvenanceRgcnMethodConfig,
    GraphRAGMethodConfig,
    PairBuildConfig,
    PrepareSplitConfig,
    ResolvedExperimentConfig,
    RgcnMethodConfig,
    SplitName,
    TwoWikiProvenanceTransformConfig,
    ranking_config,
)
from graph_memory.experiment.output import project_run_output
from graph_memory.experiment.inputs import ensure_inputs
from graph_memory.experiment.results import FinalExperimentResult
from graph_memory.experiment.tasks import (
    benchmark_retrieval_task,
    build_evidence_graphs_task,
    build_training_pairs_task,
    evaluate_rankings_task,
    generate_rankings_task,
    prefect_storage_settings,
    prepare_split_task,
    resolve_encoder_source,
    train_dense_ft_task,
    train_evidence_rgcn_task,
    train_provenance_rgcn_task,
    transform_twowiki_task,
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
                ExecutionProvenanceMethodConfig,
            ),
        ):
            test = prepare_split_task(
                source=split_sources["test"],
                config=_prepare_config(config, "test"),
            )
            assets.append(test.artifact)
            if not isinstance(method, Bm25MethodConfig):
                ranking_encoder = resolve_encoder_source(method.encoder)

        elif isinstance(method, DenseFinetuneMethodConfig):
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
            assets.extend((train.artifact, dev.artifact, test.artifact))

            encoder_source = resolve_encoder_source(method.encoder)
            effective_pairs = method.pairs
            if config.dataset.name == "twowiki_provenance":
                effective_pairs = method.pairs.model_copy(
                    update={"hard_graph_neighbor_per_positive": 0}
                )
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
                config=method.train_stage(),
                encoder_source=encoder_source,
            )
            assets.extend((pairs.artifact, model.artifact))

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
            effective = method.effective()
            encoder_source = resolve_encoder_source(effective.encoder)
            pairs = build_training_pairs_task(
                prepared=train.artifact,
                evidence_graphs=train_graphs.artifact,
                dataset=config.dataset.name,
                config=PairBuildConfig(
                    sampling=effective.pairs,
                    encoder=effective.encoder,
                    device=config.device,
                ),
                encoder_source=encoder_source,
            )
            model = train_evidence_rgcn_task(
                train_prepared=train.artifact,
                train_graphs=train_graphs.artifact,
                train_pairs=pairs.artifact,
                dev_prepared=dev.artifact,
                dev_graphs=dev_graphs.artifact,
                seed_model=None,
                dataset=config.dataset.name,
                config=method.train_stage(),
                encoder_source=encoder_source,
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
                config=method.seed.train_stage(),
                encoder_source=seed_source,
            )

            rgcn = method.effective_rgcn()
            rgcn_source = resolve_encoder_source(rgcn.encoder)
            rgcn_pairs = build_training_pairs_task(
                prepared=train.artifact,
                evidence_graphs=train_graphs.artifact,
                dataset=config.dataset.name,
                config=PairBuildConfig(
                    sampling=rgcn.pairs,
                    encoder=rgcn.encoder,
                    device=config.device,
                ),
                encoder_source=rgcn_source,
            )
            model = train_evidence_rgcn_task(
                train_prepared=train.artifact,
                train_graphs=train_graphs.artifact,
                train_pairs=rgcn_pairs.artifact,
                dev_prepared=dev.artifact,
                dev_graphs=dev_graphs.artifact,
                seed_model=seed_model.artifact,
                dataset=config.dataset.name,
                config=method.rgcn_train_stage(),
                encoder_source=rgcn_source,
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
                    model.artifact,
                )
            )

        elif isinstance(method, ExecutionProvenanceRgcnMethodConfig):
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
            effective = method.effective()
            encoder_source = resolve_encoder_source(effective.encoder)
            pairs = build_training_pairs_task(
                prepared=train.artifact,
                evidence_graphs=None,
                dataset=config.dataset.name,
                config=PairBuildConfig(
                    sampling=effective.pairs,
                    encoder=effective.encoder,
                    device=config.device,
                ),
                encoder_source=encoder_source,
            )
            model = train_provenance_rgcn_task(
                train_prepared=train.artifact,
                train_pairs=pairs.artifact,
                dev_prepared=dev.artifact,
                dataset=config.dataset.name,
                config=method.train_stage(),
                encoder_source=encoder_source,
            )
            assets.extend(
                (
                    train.artifact,
                    dev.artifact,
                    test.artifact,
                    pairs.artifact,
                    model.artifact,
                )
            )

        else:
            raise ValueError(f"unsupported final method={type(method).__name__}")

        rank_config = ranking_config(method)
        ranking = generate_rankings_task(
            prepared=test.artifact,
            evidence_graphs=ranking_graphs,
            model=None if model is None else model.artifact,
            dataset=config.dataset.name,
            method=rank_config,
            top_k=config.top_k,
            encoder_source=ranking_encoder,
            device=config.device,
        )
        evaluation = evaluate_rankings_task(
            predictions=ranking.artifact,
            prepared=test.artifact,
            evidence_graphs=ranking_graphs,
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
    if config.dataset.name == "twowiki_provenance":
        return _transform_split_sources(config)
    return {split: _direct_split_source(config, split) for split in _SPLIT_NAMES}


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


def _transform_split_sources(
    config: ResolvedExperimentConfig,
) -> dict[SplitName, FileSourceRef]:
    transform = config.dataset.transform
    if transform is None:
        raise ValueError(
            "twowiki_provenance requires a resolved transform configuration"
        )
    train_source = _direct_split_source(config, "train")
    dev_source = _direct_split_source(config, "dev")
    encoder_source = _transform_encoder_source(transform)
    result = transform_twowiki_task(
        train_source=train_source,
        dev_source=dev_source,
        config=transform,
        encoder_source=encoder_source,
    )
    return {"train": result.train, "dev": result.dev, "test": result.test}


def _transform_encoder_source(config: TwoWikiProvenanceTransformConfig):
    if config.edge_scorer not in {"dense", "hybrid"}:
        return None
    return resolve_encoder_source(
        DenseEncoderConfig(
            model_name=config.dense_model,
            query_prefix=config.dense_query_prefix,
            passage_prefix=config.dense_passage_prefix,
            batch_size=config.dense_batch_size,
        )
    )


def _prepare_config(
    config: ResolvedExperimentConfig,
    split: SplitName,
) -> PrepareSplitConfig:
    split_config = config.dataset.splits[split]
    return PrepareSplitConfig(
        dataset=config.dataset.name,
        split=split,
        count=split_config.count,
        offset=split_config.offset,
        seed=config.seed,
        strict_invalid_examples=config.dataset.strict_invalid_examples,
    )


def _unique_assets(values: list[ArtifactRef]) -> tuple[ArtifactRef, ...]:
    return tuple({(value.kind, value.digest): value for value in values}.values())


__all__ = ["run_experiment"]

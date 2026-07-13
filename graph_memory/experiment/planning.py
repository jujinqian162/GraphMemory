from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from graph_memory.experiment.config import (
    AliasArtifactRef,
    ArtifactBinding,
    ArtifactRef,
    ArtifactKind,
    Bm25GraphRerankMethodConfig,
    Bm25MethodConfig,
    DenseFinetuneMethodConfig,
    DenseFtRgcnMethodConfig,
    DenseGraphRerankMethodConfig,
    DenseMethodConfig,
    DenseRgcnMethodConfig,
    MethodConfig,
    MemoryStreamMethodConfig,
    PublicStageName,
    ResolvedExperimentConfig,
    ResolvedImportanceSplitConfig,
    SplitName,
)
from graph_memory.experiment.layout import RunLayout
from graph_memory.experiment.invocation import StageInvocation
from graph_memory.experiment.stage_status import inspect_invocation_status
from graph_memory.experiment.stage_models import (
    AblationAggregateStageConfig,
    AblationSelection,
    Bm25GraphRerankTuneStageConfig,
    Bm25GraphRerankRetrieveStageConfig,
    Bm25RetrieveStageConfig,
    DenseFinetuneRetrieveStageConfig,
    DenseFinetuneTrainStageConfig,
    DenseGraphRerankTuneStageConfig,
    DenseGraphRerankRetrieveStageConfig,
    DenseRetrieveStageConfig,
    EvaluateStageConfig,
    GraphStageConfig,
    ImportancePrepareStageConfig,
    MemoryStreamRetrieveStageConfig,
    MemoryStreamTuneStageConfig,
    OrdinaryAggregateStageConfig,
    OrdinaryRgcnTrainStageConfig,
    PairOutputs,
    PairStageConfig,
    PrepareOutputs,
    RawPrepareStageConfig,
    RetrieveStageConfig,
    RgcnRetrieveStageConfig,
    SeededRgcnTrainStageConfig,
    StageConfig,
    TrainStageConfig,
)
from graph_memory.registry.ablations import (
    ABLATION_SUITE_PATCHES,
    ExecutableAblationVariant,
    PairSamplingPatch,
    RgcnModelPatch,
)
from graph_memory.registry import Registry
from graph_memory.registry.methods import (
    ArtifactKind as RegistryArtifactKind,
)
from graph_memory.registry.retrieval import RetrievalMethodId

STAGE_ORDER: tuple[PublicStageName, ...] = (
    "prepare",
    "graphs",
    "pairs",
    "tune",
    "train",
    "retrieve",
    "evaluate",
    "aggregate",
)


@dataclass(frozen=True)
class WorkflowPlan:
    invocations: tuple[StageInvocation, ...]
    aliases: tuple[StageAlias, ...] = ()
    ablation_selections: tuple[AblationSelection, ...] = ()


@dataclass(frozen=True)
class StageAlias:
    identifier: str
    stage: PublicStageName
    method: RetrievalMethodId
    variant: str
    artifact: AliasArtifactRef
    source_invocation: StageInvocation


class _StageInvocationFactory:
    def __init__(self, config: ResolvedExperimentConfig, layout: RunLayout) -> None:
        self.config = config
        self.layout = layout
        self.methods = tuple(config.methods)

    def _prepare_invocation(self, split: SplitName) -> StageInvocation:
        split_config = self.config.dataset.splits[split]
        paths = self.layout.inputs(split)
        summary = self.layout.summary_for(paths["input"], stage="prepare")
        outputs = PrepareOutputs(
            input=paths["input"].resolve(),
            labels=paths["labels"].resolve(),
            combined=paths["combined"].resolve(),
        )
        if split_config.kind == "importance":
            if self.config.dataset.name != "hotpotqa":
                raise ValueError("importance splits are only supported for hotpotqa")
            if RetrievalMethodId.MEMORY_STREAM not in self.methods:
                raise ValueError(
                    "importance splits require selected method memory_stream"
                )
            config: StageConfig = ImportancePrepareStageConfig(
                stage="prepare",
                kind="importance",
                dataset="hotpotqa",
                split=cast(Literal["dev", "test"], split),
                canonical_inputs=split_config.source,
                canonical_labels=split_config.labels_source,
                importance=split_config.importance_path,
                outputs=outputs,
                count=split_config.count,
                offset=split_config.offset,
            )
            inputs = (
                _external("canonical_inputs", split_config.source),
                _external("canonical_labels", split_config.labels_source),
                _external("importance", split_config.importance_path),
            )
        else:
            config = RawPrepareStageConfig(
                stage="prepare",
                kind="raw",
                dataset=self.config.dataset.name,
                split=split,
                source=split_config.source,
                outputs=outputs,
                count=split_config.count,
                seed=self.config.seed,
                offset=split_config.offset,
                strict_invalid_examples=False,
            )
            inputs = (_external("raw", split_config.source),)
        output_refs = tuple(
            self.layout.artifact(role=role, path=path)
            for role, path in (
                ("inputs", paths["input"]),
                ("labels", paths["labels"]),
                ("combined", paths["combined"]),
            )
        )
        return self._invocation(
            stage="prepare",
            split=split,
            script=self.config.dataset.prepare_script,
            config=config,
            summary_path=summary.resolve(),
            inputs=inputs,
            outputs=output_refs,
            dependencies=(),
        )

    def _graph_invocation(self, split: SplitName) -> StageInvocation:
        tasks = self.layout.inputs(split)["input"].resolve()
        output = self.layout.graph(split).resolve()
        summary = self.layout.summary_for(output, stage="graphs").resolve()
        config = GraphStageConfig(
            stage="graphs",
            dataset=self.config.dataset.name,
            split=split,
            tasks=tasks,
            output=output,
            graph=self.config.graph,
        )
        return self._invocation(
            stage="graphs",
            split=split,
            script=self.layout.repository_root / "scripts" / "build_graphs.py",
            config=config,
            summary_path=summary,
            inputs=(self.layout.artifact(role="inputs", path=tasks),),
            outputs=(self.layout.artifact(role="graphs", path=output),),
            dependencies=(_identifier("prepare", split=split),),
        )

    def _pair_invocation(
        self,
        method: RetrievalMethodId,
        *,
        variant: str | None = None,
        method_config: MethodConfig | None = None,
    ) -> StageInvocation:
        config_method = method_config or self.config.method_configs.get(method)
        if not isinstance(
            config_method,
            (DenseRgcnMethodConfig, DenseFinetuneMethodConfig, DenseFtRgcnMethodConfig),
        ):
            raise TypeError(
                f"pair stage requires trainable method config: {method.value}"
            )
        tasks = self.layout.inputs("train")["input"].resolve()
        labels = self.layout.inputs("train")["labels"].resolve()
        graphs = self.layout.graph("train").resolve()
        pairs = self.layout.train_pairs(method, variant=variant).resolve()
        pair_summary = self.layout.train_pair_summary(method, variant=variant).resolve()
        summary = self.layout.summary_for(pairs, stage="pairs").resolve()
        config = PairStageConfig(
            stage="pairs",
            dataset=self.config.dataset.name,
            method=cast(
                Literal[
                    "dense_rgcn_graph_retriever",
                    "dense_ft",
                    "dense_ft_rgcn_graph_retriever",
                ],
                method.value,
            ),
            variant=variant,
            tasks=tasks,
            labels=labels,
            graphs=graphs,
            outputs=PairOutputs(
                pairs=pairs,
                pair_summary=pair_summary,
            ),
            sampling=config_method.pairs,
            hard_dense_encoder=config_method.encoder,
            device=config_method.train.trainer.device,
        )
        return self._invocation(
            stage="pairs",
            method=method,
            variant=variant,
            script=self.layout.repository_root / "scripts" / "build_train_pairs.py",
            config=config,
            summary_path=summary,
            inputs=(
                self.layout.artifact(role="inputs", path=tasks),
                self.layout.artifact(role="labels", path=labels),
                self.layout.artifact(role="graphs", path=graphs),
            ),
            outputs=(
                self.layout.artifact(role="train_pairs", path=pairs),
                self.layout.artifact(role="train_pair_summary", path=pair_summary),
            ),
            dependencies=(
                _identifier("prepare", split="train"),
                _identifier("graphs", split="train"),
            ),
        )

    def _tune_invocation(self, method: RetrievalMethodId) -> StageInvocation:
        method_config = self.config.method_configs.get(method)
        tasks = self.layout.inputs("dev")["input"].resolve()
        labels = self.layout.inputs("dev")["labels"].resolve()
        graphs = self.layout.graph("dev").resolve()
        selected = self.layout.tuned(method).resolve()
        candidates = self.layout.tuned_candidates(method).resolve()
        summary = self.layout.summary_for(selected, stage="tune").resolve()
        inputs: tuple[ArtifactBinding, ...]
        if isinstance(method_config, Bm25GraphRerankMethodConfig):
            config: StageConfig = Bm25GraphRerankTuneStageConfig(
                stage="tune",
                dataset=self.config.dataset.name,
                method="bm25_graph_rerank",
                tasks=tasks,
                labels=labels,
                graphs=graphs,
                selected_config=selected,
                candidates=candidates,
                top_k=self.config.top_k,
                search_space=self.config.search_spaces.graph_rerank,
            )
            inputs = (
                self.layout.artifact(role="inputs", path=tasks),
                self.layout.artifact(role="labels", path=labels),
                self.layout.artifact(role="graphs", path=graphs),
            )
        elif isinstance(method_config, DenseGraphRerankMethodConfig):
            config = DenseGraphRerankTuneStageConfig(
                stage="tune",
                dataset=self.config.dataset.name,
                method="dense_graph_rerank",
                tasks=tasks,
                labels=labels,
                graphs=graphs,
                selected_config=selected,
                candidates=candidates,
                top_k=self.config.top_k,
                encoder=method_config.encoder,
                search_space=self.config.search_spaces.graph_rerank,
            )
            inputs = (
                self.layout.artifact(role="inputs", path=tasks),
                self.layout.artifact(role="labels", path=labels),
                self.layout.artifact(role="graphs", path=graphs),
            )
        elif isinstance(method_config, MemoryStreamMethodConfig):
            if self.config.dataset.name != "hotpotqa":
                raise ValueError("Memory Stream is only supported for hotpotqa")
            importance_split = self.config.dataset.splits["dev"]
            if not isinstance(importance_split, ResolvedImportanceSplitConfig):
                raise ValueError(
                    "Memory Stream requires dataset=hotpotqa-memory-stream"
                )
            config = MemoryStreamTuneStageConfig(
                stage="tune",
                kind="memory_stream",
                dataset="hotpotqa",
                method="memory_stream",
                tasks=tasks,
                labels=labels,
                graphs=graphs,
                importance=importance_split.importance_path,
                selected_config=selected,
                candidates=candidates,
                top_k=self.config.top_k,
                encoder=method_config.encoder,
                search_space=self.config.search_spaces.memory_stream,
            )
            inputs = (
                self.layout.artifact(role="inputs", path=tasks),
                self.layout.artifact(role="labels", path=labels),
                self.layout.artifact(role="graphs", path=graphs),
                _external("importance", importance_split.importance_path),
            )
        else:
            raise TypeError(f"method has no tune config: {method.value}")
        return self._invocation(
            stage="tune",
            method=method,
            script=(
                self.layout.repository_root
                / "scripts"
                / (
                    "tune_memory_stream.py"
                    if method is RetrievalMethodId.MEMORY_STREAM
                    else "tune_graph_rerank.py"
                )
            ),
            config=config,
            summary_path=summary,
            inputs=inputs,
            outputs=(
                self.layout.artifact(role="selected_config", path=selected),
                self.layout.artifact(role="candidate_table", path=candidates),
            ),
            dependencies=(
                _identifier("prepare", split="dev"),
                _identifier("graphs", split="dev"),
            ),
        )

    def _train_invocation(
        self,
        method: RetrievalMethodId,
        *,
        variant: str | None = None,
        method_config: MethodConfig | None = None,
        pair_variant: str | None = None,
    ) -> StageInvocation:
        config_method = method_config or self.config.method_configs.get(method)
        learned_root = self.layout.learned_root(method, variant=variant).resolve()
        metrics = self.layout.training_metrics(method, variant=variant).resolve()
        pair_path = self.layout.train_pairs(method, variant=pair_variant).resolve()
        spec = Registry.methods.get(method)
        if spec.train_artifact is None:
            raise TypeError(f"train stage requires trainable method: {method.value}")
        kind: ArtifactKind = (
            "file"
            if spec.train_artifact.kind is RegistryArtifactKind.FILE
            else "directory"
        )
        checkpoint = self.layout.checkpoint(
            method, kind=kind, variant=variant
        ).resolve()
        summary = self.layout.summary_for(checkpoint, stage="train").resolve()
        dependencies = [
            _identifier("pairs", method=method, variant=pair_variant),
            _identifier("prepare", split="dev"),
        ]
        if method is RetrievalMethodId.DENSE_FT_RGCN_GRAPH_RETRIEVER:
            dependencies.append(_identifier("train", method=RetrievalMethodId.DENSE_FT))
        if isinstance(config_method, (DenseRgcnMethodConfig, DenseFtRgcnMethodConfig)):
            dependencies.append(_identifier("graphs", split="dev"))
            seed_checkpoint = (
                self.layout.checkpoint(
                    RetrievalMethodId.DENSE_FT,
                    kind="directory",
                ).resolve()
                if method is RetrievalMethodId.DENSE_FT_RGCN_GRAPH_RETRIEVER
                else None
            )
            if seed_checkpoint is not None:
                config: TrainStageConfig = SeededRgcnTrainStageConfig(
                    stage="train",
                    method="dense_ft_rgcn_graph_retriever",
                    variant=variant,
                    dataset=self.config.dataset.name,
                    train_tasks=self.layout.inputs("train")["input"].resolve(),
                    train_labels=self.layout.inputs("train")["labels"].resolve(),
                    train_graphs=self.layout.graph("train").resolve(),
                    train_pairs=pair_path,
                    dev_tasks=self.layout.inputs("dev")["input"].resolve(),
                    dev_labels=self.layout.inputs("dev")["labels"].resolve(),
                    dev_graphs=self.layout.graph("dev").resolve(),
                    output_dir=learned_root,
                    checkpoint_dir=checkpoint.parent,
                    metrics=metrics,
                    seed_model_dir=seed_checkpoint,
                    encoder=config_method.encoder,
                    pairs=config_method.pairs,
                    train=config_method.train,
                )
            else:
                config = OrdinaryRgcnTrainStageConfig(
                    stage="train",
                    method="dense_rgcn_graph_retriever",
                    variant=variant,
                    dataset=self.config.dataset.name,
                    train_tasks=self.layout.inputs("train")["input"].resolve(),
                    train_labels=self.layout.inputs("train")["labels"].resolve(),
                    train_graphs=self.layout.graph("train").resolve(),
                    train_pairs=pair_path,
                    dev_tasks=self.layout.inputs("dev")["input"].resolve(),
                    dev_labels=self.layout.inputs("dev")["labels"].resolve(),
                    dev_graphs=self.layout.graph("dev").resolve(),
                    output_dir=learned_root,
                    checkpoint_dir=checkpoint.parent,
                    metrics=metrics,
                    encoder=config_method.encoder,
                    pairs=config_method.pairs,
                    train=config_method.train,
                )
            inputs = (
                self.layout.artifact(
                    role="inputs", path=self.layout.inputs("train")["input"]
                ),
                self.layout.artifact(
                    role="labels", path=self.layout.inputs("train")["labels"]
                ),
                self.layout.artifact(role="graphs", path=self.layout.graph("train")),
                self.layout.artifact(role="train_pairs", path=pair_path),
                self.layout.artifact(
                    role="dev_inputs", path=self.layout.inputs("dev")["input"]
                ),
                self.layout.artifact(
                    role="dev_labels", path=self.layout.inputs("dev")["labels"]
                ),
                self.layout.artifact(role="dev_graphs", path=self.layout.graph("dev")),
            )
            if seed_checkpoint is not None:
                inputs = (
                    *inputs,
                    self.layout.artifact(
                        role="seed_checkpoint", path=seed_checkpoint, kind="directory"
                    ),
                )
        elif isinstance(config_method, DenseFinetuneMethodConfig):
            config = DenseFinetuneTrainStageConfig(
                stage="train",
                method="dense_ft",
                variant=variant,
                dataset=self.config.dataset.name,
                train_tasks=self.layout.inputs("train")["input"].resolve(),
                train_labels=self.layout.inputs("train")["labels"].resolve(),
                train_pairs=pair_path,
                dev_tasks=self.layout.inputs("dev")["input"].resolve(),
                dev_labels=self.layout.inputs("dev")["labels"].resolve(),
                output_dir=learned_root,
                model_dir=checkpoint,
                metrics=metrics,
                encoder=config_method.encoder,
                pairs=config_method.pairs,
                train=config_method.train,
            )
            inputs = (
                self.layout.artifact(
                    role="inputs", path=self.layout.inputs("train")["input"]
                ),
                self.layout.artifact(
                    role="labels", path=self.layout.inputs("train")["labels"]
                ),
                self.layout.artifact(role="train_pairs", path=pair_path),
                self.layout.artifact(
                    role="dev_inputs", path=self.layout.inputs("dev")["input"]
                ),
                self.layout.artifact(
                    role="dev_labels", path=self.layout.inputs("dev")["labels"]
                ),
            )
        else:
            raise TypeError(f"unsupported train config: {type(config_method).__name__}")
        return self._invocation(
            stage="train",
            method=method,
            variant=variant,
            script=self.layout.repository_root / "scripts" / "train_method.py",
            config=config,
            summary_path=summary,
            inputs=inputs,
            outputs=(
                self.layout.artifact(role="checkpoint", path=checkpoint, kind=kind),
                self.layout.artifact(role="train_metrics", path=metrics),
            ),
            dependencies=tuple(dependencies),
        )

    def _retrieve_invocation(
        self,
        method: RetrievalMethodId,
        *,
        variant: str | None = None,
        method_config: MethodConfig | None = None,
    ) -> StageInvocation:
        config_method = method_config or self.config.method_configs.get(method)
        tasks = self.layout.inputs("test")["input"].resolve()
        output = self.layout.prediction(method, variant=variant).resolve()
        summary = self.layout.summary_for(output, stage="retrieve").resolve()
        dependencies = [_identifier("prepare", split="test")]
        inputs: list[ArtifactBinding] = [
            self.layout.artifact(role="inputs", path=tasks)
        ]
        if isinstance(config_method, Bm25MethodConfig):
            config: RetrieveStageConfig = Bm25RetrieveStageConfig(
                stage="retrieve",
                method="bm25",
                variant=variant,
                dataset=self.config.dataset.name,
                tasks=tasks,
                output=output,
                top_k=self.config.top_k,
            )
        elif isinstance(config_method, DenseMethodConfig):
            config = DenseRetrieveStageConfig(
                stage="retrieve",
                method="dense",
                variant=variant,
                dataset=self.config.dataset.name,
                tasks=tasks,
                output=output,
                top_k=self.config.top_k,
                encoder=config_method.encoder,
            )
        elif isinstance(config_method, MemoryStreamMethodConfig):
            importance_split = self.config.dataset.splits["test"]
            if not isinstance(importance_split, ResolvedImportanceSplitConfig):
                raise ValueError(
                    "Memory Stream requires dataset=hotpotqa-memory-stream"
                )
            selected = self.layout.tuned(method).resolve()
            config = MemoryStreamRetrieveStageConfig(
                method="memory_stream",
                encoder=config_method.encoder,
                selected_config=selected,
                importance=importance_split.importance_path,
                scoring=config_method.scoring,
                capped_test_count=self.config.dataset.splits["test"].count,
                stage="retrieve",
                variant=variant,
                dataset="hotpotqa",
                tasks=tasks,
                output=output,
                top_k=self.config.top_k,
            )
            inputs.extend(
                (
                    self.layout.artifact(role="selected_config", path=selected),
                    _external("importance", importance_split.importance_path),
                )
            )
            dependencies.append(_identifier("tune", method=method))
        elif isinstance(config_method, Bm25GraphRerankMethodConfig):
            graphs = self.layout.graph("test").resolve()
            selected = self.layout.tuned(method).resolve()
            config = Bm25GraphRerankRetrieveStageConfig(
                method="bm25_graph_rerank",
                graphs=graphs,
                selected_config=selected,
                seed_method="bm25",
                stage="retrieve",
                variant=variant,
                dataset=self.config.dataset.name,
                tasks=tasks,
                output=output,
                top_k=self.config.top_k,
            )
            inputs.extend(
                (
                    self.layout.artifact(role="graphs", path=graphs),
                    self.layout.artifact(role="selected_config", path=selected),
                )
            )
            dependencies.extend(
                (
                    _identifier("graphs", split="test"),
                    _identifier("tune", method=method),
                )
            )
        elif isinstance(config_method, DenseGraphRerankMethodConfig):
            graphs = self.layout.graph("test").resolve()
            selected = self.layout.tuned(method).resolve()
            config = DenseGraphRerankRetrieveStageConfig(
                method="dense_graph_rerank",
                graphs=graphs,
                selected_config=selected,
                seed_method="dense",
                encoder=config_method.encoder,
                stage="retrieve",
                variant=variant,
                dataset=self.config.dataset.name,
                tasks=tasks,
                output=output,
                top_k=self.config.top_k,
            )
            inputs.extend(
                (
                    self.layout.artifact(role="graphs", path=graphs),
                    self.layout.artifact(role="selected_config", path=selected),
                )
            )
            dependencies.extend(
                (
                    _identifier("graphs", split="test"),
                    _identifier("tune", method=method),
                )
            )
        elif isinstance(
            config_method, (DenseRgcnMethodConfig, DenseFtRgcnMethodConfig)
        ):
            graphs = self.layout.graph("test").resolve()
            checkpoint = self.layout.checkpoint(
                method, kind="file", variant=variant
            ).resolve()
            config = RgcnRetrieveStageConfig(
                method=cast(
                    Literal[
                        "dense_rgcn_graph_retriever",
                        "dense_ft_rgcn_graph_retriever",
                    ],
                    method.value,
                ),
                graphs=graphs,
                checkpoint=checkpoint,
                device=config_method.train.trainer.device,
                stage="retrieve",
                variant=variant,
                dataset=self.config.dataset.name,
                tasks=tasks,
                output=output,
                top_k=self.config.top_k,
            )
            inputs.extend(
                (
                    self.layout.artifact(role="graphs", path=graphs),
                    self.layout.artifact(role="checkpoint", path=checkpoint),
                )
            )
            dependencies.extend(
                (
                    _identifier("graphs", split="test"),
                    _identifier("train", method=method, variant=variant),
                )
            )
        elif isinstance(config_method, DenseFinetuneMethodConfig):
            model_dir = self.layout.checkpoint(
                method, kind="directory", variant=variant
            ).resolve()
            config = DenseFinetuneRetrieveStageConfig(
                method="dense_ft",
                model_dir=model_dir,
                device=config_method.train.trainer.device,
                stage="retrieve",
                variant=variant,
                dataset=self.config.dataset.name,
                tasks=tasks,
                output=output,
                top_k=self.config.top_k,
            )
            inputs.append(
                self.layout.artifact(
                    role="checkpoint", path=model_dir, kind="directory"
                )
            )
            dependencies.append(_identifier("train", method=method, variant=variant))
        else:
            raise TypeError(
                f"unsupported retrieve config: {type(config_method).__name__}"
            )
        return self._invocation(
            stage="retrieve",
            method=method,
            variant=variant,
            script=self.layout.repository_root / "scripts" / "run_retrieval.py",
            config=config,
            summary_path=summary,
            inputs=tuple(inputs),
            outputs=(self.layout.artifact(role="predictions", path=output),),
            dependencies=tuple(dependencies),
        )

    def _evaluate_invocation(
        self,
        method: RetrievalMethodId,
        *,
        variant: str | None = None,
    ) -> StageInvocation:
        predictions = self.layout.prediction(method, variant=variant).resolve()
        labels = self.layout.inputs("test")["labels"].resolve()
        graphs = self.layout.graph("test").resolve()
        metrics = self.layout.metric(method, variant=variant).resolve()
        failures = self.layout.failure_cases(method, variant=variant).resolve()
        config = EvaluateStageConfig(
            stage="evaluate",
            dataset=self.config.dataset.name,
            method=method.value,
            variant=variant,
            predictions=predictions,
            labels=labels,
            graphs=graphs,
            metrics=metrics,
            failure_cases=failures,
            failure_case_limit=50,
            top_k=self.config.top_k,
        )
        return self._invocation(
            stage="evaluate",
            method=method,
            variant=variant,
            script=self.layout.repository_root / "scripts" / "evaluate_retrieval.py",
            config=config,
            summary_path=self.layout.summary_for(metrics, stage="evaluate").resolve(),
            inputs=(
                self.layout.artifact(role="predictions", path=predictions),
                self.layout.artifact(role="labels", path=labels),
                self.layout.artifact(role="graphs", path=graphs),
            ),
            outputs=(
                self.layout.artifact(role="metrics", path=metrics),
                self.layout.artifact(role="failure_cases", path=failures),
            ),
            dependencies=(
                _identifier("retrieve", method=method, variant=variant),
                _identifier("prepare", split="test"),
                _identifier("graphs", split="test"),
            ),
        )

    def _aggregate_invocation(
        self,
        selections: tuple[AblationSelection, ...],
    ) -> StageInvocation:
        main = self.layout.table("main").resolve()
        path = self.layout.table("path").resolve()
        efficiency = self.layout.table("efficiency").resolve()
        ablation = self.layout.table("ablation").resolve() if selections else None
        ablation_metrics = [
            self.layout.metric(selection.method, variant=selection.variant).resolve()
            for selection in selections
            if selection.variant != "full_rgcn"
        ]
        metric_paths = [self.layout.metric(method).resolve() for method in self.methods]
        summary = self.layout.summary_for(main, stage="aggregate").resolve()
        config: StageConfig = (
            AblationAggregateStageConfig(
                stage="aggregate",
                kind="ablation",
                metrics=metric_paths,
                main=main,
                path=path,
                efficiency=efficiency,
                ablation_metrics=ablation_metrics,
                ablation_index=self.layout.ablation_metrics_index.resolve(),
                ablation=self.layout.table("ablation").resolve(),
                ablation_selections=list(selections),
            )
            if selections
            else OrdinaryAggregateStageConfig(
                stage="aggregate",
                kind="ordinary",
                metrics=metric_paths,
                main=main,
                path=path,
                efficiency=efficiency,
            )
        )
        metrics = tuple(
            self.layout.artifact(role="metrics", path=self.layout.metric(method))
            for method in self.methods
        )
        inputs = list(metrics)
        dependencies = [
            _identifier("evaluate", method=method) for method in self.methods
        ]
        if selections:
            inputs.append(
                self.layout.artifact(
                    role="ablation_index",
                    path=self.layout.ablation_metrics_index,
                )
            )
            executable_selections = tuple(
                value for value in selections if value.variant != "full_rgcn"
            )
            for selection, metric_path in zip(
                executable_selections,
                ablation_metrics,
                strict=True,
            ):
                inputs.append(
                    self.layout.artifact(
                        role="ablation_metrics",
                        path=metric_path,
                    )
                )
                dependencies.append(
                    _identifier(
                        "evaluate",
                        method=selection.method,
                        variant=selection.variant,
                    )
                )
        outputs = [
            self.layout.artifact(role="main_table", path=main),
            self.layout.artifact(role="path_table", path=path),
            self.layout.artifact(role="efficiency_table", path=efficiency),
        ]
        if ablation is not None:
            outputs.append(self.layout.artifact(role="ablation_table", path=ablation))
        return self._invocation(
            stage="aggregate",
            script=self.layout.repository_root / "scripts" / "aggregate_tables.py",
            config=config,
            summary_path=summary,
            inputs=tuple(inputs),
            outputs=tuple(outputs),
            dependencies=tuple(dict.fromkeys(dependencies)),
        )

    def _variant_invocations(
        self,
        method: RetrievalMethodId,
        variant: str,
        ordinary_by_id: dict[str, StageInvocation],
    ) -> tuple[tuple[StageInvocation, ...], tuple[StageAlias, ...]]:
        config_method = self.config.method_configs.get(method)
        if not isinstance(
            config_method, (DenseRgcnMethodConfig, DenseFtRgcnMethodConfig)
        ):
            raise ValueError(f"method does not support R-GCN ablations: {method.value}")
        patched, invalidated_from = _apply_rgcn_ablation(config_method, variant)
        invocations: list[StageInvocation] = []
        aliases: list[StageAlias] = []
        pair_variant: str | None
        if invalidated_from == "pairs":
            pair_variant = variant
            invocations.append(
                self._pair_invocation(method, variant=variant, method_config=patched)
            )
        else:
            pair_variant = None
            aliases.append(
                StageAlias(
                    identifier=_identifier("pairs", method=method, variant=variant),
                    stage="pairs",
                    method=method,
                    variant=variant,
                    artifact=self.layout.alias_artifact(
                        role="train_pairs",
                        path=self.layout.train_pairs(method, variant=variant),
                        alias_of=self.layout.train_pairs(method).resolve(),
                    ),
                    source_invocation=ordinary_by_id[
                        _identifier("pairs", method=method)
                    ],
                )
            )
        invocations.append(
            self._train_invocation(
                method,
                variant=variant,
                method_config=patched,
                pair_variant=pair_variant,
            )
        )
        invocations.append(
            self._retrieve_invocation(method, variant=variant, method_config=patched)
        )
        invocations.append(self._evaluate_invocation(method, variant=variant))
        return tuple(invocations), tuple(aliases)

    def _baseline_aliases(
        self,
        method: RetrievalMethodId,
        ordinary_by_id: dict[str, StageInvocation],
    ) -> tuple[StageAlias, ...]:
        train_artifact = Registry.methods.get(method).train_artifact
        if train_artifact is None:
            raise ValueError(f"ablation method has no checkpoint kind: {method.value}")
        checkpoint_kind: ArtifactKind = (
            "file" if train_artifact.kind is RegistryArtifactKind.FILE else "directory"
        )
        bindings: tuple[tuple[PublicStageName, Path, Path, ArtifactKind, str], ...] = (
            (
                "pairs",
                self.layout.train_pairs(method, variant="full_rgcn"),
                self.layout.train_pairs(method),
                "file",
                "train_pairs",
            ),
            (
                "train",
                self.layout.checkpoint(
                    method, kind=checkpoint_kind, variant="full_rgcn"
                ),
                self.layout.checkpoint(method, kind=checkpoint_kind),
                checkpoint_kind,
                "checkpoint",
            ),
            (
                "retrieve",
                self.layout.prediction(method, variant="full_rgcn"),
                self.layout.prediction(method),
                "file",
                "predictions",
            ),
            (
                "evaluate",
                self.layout.metric(method, variant="full_rgcn"),
                self.layout.metric(method),
                "file",
                "metrics",
            ),
        )
        return tuple(
            StageAlias(
                identifier=_identifier(stage, method=method, variant="full_rgcn"),
                stage=stage,
                method=method,
                variant="full_rgcn",
                artifact=self.layout.alias_artifact(
                    role=role,
                    path=target,
                    kind=kind,
                    alias_of=source.resolve(),
                ),
                source_invocation=ordinary_by_id[_identifier(stage, method=method)],
            )
            for stage, target, source, kind, role in bindings
        )

    def _invocation(
        self,
        *,
        stage: PublicStageName,
        script: Path,
        config: StageConfig,
        summary_path: Path,
        inputs: tuple[ArtifactBinding, ...],
        outputs: tuple[ArtifactBinding, ...],
        dependencies: tuple[str, ...],
        method: RetrievalMethodId | None = None,
        split: SplitName | None = None,
        variant: str | None = None,
    ) -> StageInvocation:
        identifier = _identifier(
            stage,
            method=method,
            split=split,
            variant=variant,
        )
        return StageInvocation(
            identifier=identifier,
            stage=stage,
            method=method,
            split=split,
            variant=variant,
            script=script.resolve(),
            config_path=self.layout.stage_config(
                stage,
                method=method,
                split=split,
                variant=variant,
            ).resolve(),
            summary_path=summary_path.resolve(),
            config=config,
            inputs=inputs,
            outputs=outputs,
            dependencies=dependencies,
        )


class WorkflowPlanner:
    def __init__(self, config: ResolvedExperimentConfig, layout: RunLayout) -> None:
        self.config = config
        self.layout = layout
        self.methods = tuple(config.methods)
        self.train_methods = Registry.methods.expand_train_dependencies(self.methods)
        self.factory = _StageInvocationFactory(config, layout)

    def build(self, *, validate_external: bool = True) -> WorkflowPlan:
        ordinary = self._ordinary_invocations()
        ordinary_by_id = {item.identifier: item for item in ordinary}
        variants = self._selected_variants()
        if not variants:
            selected = self._select_range(ordinary)
            if validate_external:
                self._validate_external_dependencies(selected, ordinary)
            return WorkflowPlan(invocations=selected)

        ordinary_without_aggregate = tuple(
            item for item in ordinary if item.stage != "aggregate"
        )
        variant_invocations: list[StageInvocation] = []
        aliases: list[StageAlias] = []
        selections: list[AblationSelection] = []

        for method in dict.fromkeys(method for method, _ in variants):
            aliases.extend(self.factory._baseline_aliases(method, ordinary_by_id))

        if self.config.ablation.only:
            self._validate_ablation_baselines(variants)
            ordinary_selected = tuple(
                item
                for item in ordinary_without_aggregate
                if item.stage in {"prepare", "graphs"}
            )
        else:
            ordinary_selected = ordinary_without_aggregate

        for method, variant in variants:
            selections.append(AblationSelection(method=method, variant="full_rgcn"))
            selections.append(AblationSelection(method=method, variant=variant))
            built, variant_aliases = self.factory._variant_invocations(
                method,
                variant,
                ordinary_by_id,
            )
            variant_invocations.extend(built)
            aliases.extend(variant_aliases)

        aggregate = self.factory._aggregate_invocation(tuple(dict.fromkeys(selections)))
        combined = (*ordinary_selected, *variant_invocations, aggregate)
        selected = self._select_range(tuple(combined))
        dependency_graph = (
            *ordinary_without_aggregate,
            *variant_invocations,
            aggregate,
        )
        if validate_external:
            self._validate_external_dependencies(selected, tuple(dependency_graph))
        return WorkflowPlan(
            invocations=selected,
            aliases=tuple(aliases),
            ablation_selections=tuple(dict.fromkeys(selections)),
        )

    def _ordinary_invocations(self) -> tuple[StageInvocation, ...]:
        split_names: tuple[SplitName, ...] = ("train", "dev", "test")
        invocations = [
            *(self.factory._prepare_invocation(split) for split in split_names),
            *(self.factory._graph_invocation(split) for split in split_names),
            *(self.factory._pair_invocation(method) for method in self.train_methods),
            *(
                self.factory._tune_invocation(method)
                for method in self.methods
                if Registry.methods.get(method).tuning is not None
            ),
            *(self.factory._train_invocation(method) for method in self.train_methods),
            *(self.factory._retrieve_invocation(method) for method in self.methods),
            *(self.factory._evaluate_invocation(method) for method in self.methods),
            self.factory._aggregate_invocation(()),
        ]
        return tuple(invocations)

    def _selected_variants(self) -> tuple[tuple[RetrievalMethodId, str], ...]:
        selection = self.config.ablation.variants
        if selection == []:
            if self.config.ablation.only:
                raise ValueError("ablation.only=true requires executable variants")
            return ()
        requested = None if selection == "all" else set(selection)
        selected: list[tuple[RetrievalMethodId, str]] = []
        supported_names: set[str] = set()
        for method in self.methods:
            suite = ABLATION_SUITE_PATCHES.get(method)
            if suite is None:
                continue
            for variant in suite.variants:
                if variant.baseline_alias:
                    continue
                supported_names.add(variant.identifier)
                if requested is None or variant.identifier in requested:
                    selected.append((method, variant.identifier))
        if requested is not None:
            missing = sorted(requested - supported_names)
            if missing:
                raise ValueError(f"unknown ablation variants: {missing}")
        if not selected:
            raise ValueError(
                "no selected method exposes the requested ablation variants"
            )
        return tuple(selected)

    def _validate_ablation_baselines(
        self,
        variants: tuple[tuple[RetrievalMethodId, str], ...],
    ) -> None:
        for method in dict.fromkeys(method for method, _ in variants):
            metrics = self.layout.metric(method)
            baseline = self.factory._evaluate_invocation(method)
            if inspect_invocation_status(baseline).state != "complete":
                raise ValueError(
                    f"ablation-only requires ordinary baseline metrics: {metrics}"
                )

    def _select_range(
        self,
        invocations: tuple[StageInvocation, ...],
    ) -> tuple[StageInvocation, ...]:
        start = (
            STAGE_ORDER.index(self.config.stages.from_stage)
            if self.config.stages.from_stage is not None
            else 0
        )
        end = (
            STAGE_ORDER.index(self.config.stages.to_stage)
            if self.config.stages.to_stage is not None
            else len(STAGE_ORDER) - 1
        )
        return tuple(
            item
            for item in invocations
            if start <= STAGE_ORDER.index(item.stage) <= end
        )

    def _validate_external_dependencies(
        self,
        selected: tuple[StageInvocation, ...],
        all_invocations: tuple[StageInvocation, ...],
    ) -> None:
        selected_ids = {item.identifier for item in selected}
        by_id = {item.identifier: item for item in all_invocations}
        for item in selected:
            for dependency in item.dependencies:
                if dependency in selected_ids:
                    continue
                dependency_item = by_id.get(dependency)
                if dependency_item is None:
                    continue
                required_paths = {input_ref.path for input_ref in item.inputs}
                for output in dependency_item.outputs:
                    if output.path not in required_paths:
                        continue
                    dependency_state = inspect_invocation_status(dependency_item).state
                    if dependency_state != "complete":
                        raise ValueError(
                            f"stage={item.identifier} requires external dependency "
                            f"role={output.role} path={output.path} "
                            f"state={dependency_state}"
                        )


def format_plan(plan: WorkflowPlan) -> str:
    blocks: list[str] = []
    for index, item in enumerate(plan.invocations, start=1):
        blocks.append(format_invocation(item, index=index))
    return "\n".join(blocks)


def format_invocation(item: StageInvocation, *, index: int) -> str:
    qualifiers = [f"stage={item.stage}"]
    if item.method is not None:
        qualifiers.append(f"method={item.method.value}")
    if item.split is not None:
        qualifiers.append(f"split={item.split}")
    if item.variant is not None:
        qualifiers.append(f"variant={item.variant}")
    heading = " ".join(qualifiers)
    heading = f"[{index}] {heading}"
    return "\n".join(
        (
            heading,
            f"script: {item.script}",
            f"config: {item.config_path}",
            "command:",
            *(f"  {token}" for token in item.argv),
        )
    )


def _identifier(
    stage: PublicStageName,
    *,
    method: RetrievalMethodId | None = None,
    split: SplitName | None = None,
    variant: str | None = None,
) -> str:
    method_name = None if method is None else method.value
    qualifier = method_name or split or "aggregate"
    base = f"{stage}:{qualifier}"
    return f"{base}:{variant}" if variant is not None else base


def _external(
    role: str,
    path: Path,
) -> ArtifactRef:
    return ArtifactRef(
        role=role,
        path=path.resolve(),
        kind="file",
    )


def _apply_rgcn_ablation(
    config: DenseRgcnMethodConfig | DenseFtRgcnMethodConfig,
    variant: str,
) -> tuple[DenseRgcnMethodConfig | DenseFtRgcnMethodConfig, PublicStageName]:
    suite = ABLATION_SUITE_PATCHES[RetrievalMethodId(config.method)]
    patch = next((item for item in suite.variants if item.identifier == variant), None)
    if not isinstance(patch, ExecutableAblationVariant):
        raise ValueError(f"unknown executable ablation variant={variant!r}")
    config_patch = patch.config_patch
    if isinstance(config_patch, PairSamplingPatch):
        pairs = config.pairs.model_copy(update=config_patch.updates())
        return (
            config.model_copy(update={"pairs": pairs}),
            patch.earliest_invalidated_stage,
        )
    if not isinstance(config_patch, RgcnModelPatch):
        raise TypeError(
            f"unsupported R-GCN ablation patch: {type(config_patch).__name__}"
        )
    model = config.train.model.model_copy(update=config_patch.updates())
    train = config.train.model_copy(update={"model": model})
    return (
        config.model_copy(update={"train": train}),
        patch.earliest_invalidated_stage,
    )


__all__ = [
    "STAGE_ORDER",
    "StageAlias",
    "WorkflowPlan",
    "WorkflowPlanner",
    "format_plan",
    "format_invocation",
]

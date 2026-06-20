from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

from graph_memory.contracts.training_pairs import TrainPairRecord
from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.retrieval.requests import TextRankingRequest
from graph_memory.embeddings import load_sentence_transformer
from graph_memory.models.dense_finetune.contracts import DenseFinetuneDataSettings, DenseFinetuneIREvaluatorPayload
from graph_memory.models.dense_finetune.data import build_dense_finetune_examples, build_ir_evaluator_payload
from graph_memory.models.dense_finetune.metadata import (
    DenseFinetuneModelMetadata,
    DenseFinetuneSelectionMetadata,
    write_dense_ft_model_metadata,
)


@dataclass(frozen=True)
class DenseFinetuneTrainerSettings:
    learning_rate: float = 2e-5
    train_batch_size: int = 16
    eval_batch_size: int = 64
    epochs: int = 1
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    random_seed: int = 13
    device: str = "cuda"
    use_amp: bool = False


@dataclass(frozen=True)
class DenseFinetuneSelectionSettings:
    best_metric: str = "eval_dev_cos_sim_map@100"
    higher_is_better: bool = True


@dataclass(frozen=True)
class DenseFinetuneRunConfig:
    base_model: str
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    batch_size: int = 64
    data: DenseFinetuneDataSettings = DenseFinetuneDataSettings()
    trainer: DenseFinetuneTrainerSettings = DenseFinetuneTrainerSettings()
    selection: DenseFinetuneSelectionSettings = DenseFinetuneSelectionSettings()


class DenseFinetuneModel(Protocol):
    def fit(self, **kwargs: object) -> None:
        ...

    def save(self, output_path: str) -> None:
        ...


class DenseFinetuneTrainer(Protocol):
    def train(self) -> None:
        ...

    @property
    def metric_records(self) -> Sequence[Mapping[str, object]]:
        ...


@dataclass(frozen=True)
class DenseFinetuneTrainerRequest:
    model: DenseFinetuneModel
    train_rows: tuple[dict[str, str], ...]
    evaluator_payload: DenseFinetuneIREvaluatorPayload
    config: DenseFinetuneRunConfig
    output_dir: Path
    model_dir: Path


@dataclass(frozen=True)
class DenseFinetuneTrainingResult:
    model_dir: Path
    metadata_path: Path
    metric_records: tuple[dict[str, object], ...]
    selected_metric_name: str
    selected_metric_value: float | None


DenseFinetuneModelFactory = Callable[[str, str], DenseFinetuneModel]
DenseFinetuneTrainerFactory = Callable[[DenseFinetuneTrainerRequest], DenseFinetuneTrainer]


@dataclass(frozen=True)
class DenseFinetuneTrainingComponents:
    InputExample: Any
    DataLoader: Any
    InformationRetrievalEvaluator: Any
    MultipleNegativesRankingLoss: Any


@dataclass
class _DenseFinetuneMetricTracker:
    selected_metric_name: str
    higher_is_better: bool
    _records: list[dict[str, object]] = field(default_factory=list)
    _best_epoch: int | None = None
    _best_metric: float | None = None

    @property
    def records(self) -> tuple[dict[str, object], ...]:
        return tuple(dict(record) for record in self._records)

    @property
    def best_metric(self) -> float | None:
        return self._best_metric

    def record(
        self,
        *,
        epoch: int,
        global_step: int,
        train_loss: float | None,
        selected_metric_value: float,
    ) -> bool:
        metric = float(selected_metric_value)
        improved = self._best_metric is None or self._is_improvement(metric, self._best_metric)
        if improved:
            self._best_epoch = epoch
            self._best_metric = metric
        if self._best_epoch is None or self._best_metric is None:
            raise RuntimeError("dense-ft metric tracker has no best metric after recording a row.")
        self._records.append(
            {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                self.selected_metric_name: metric,
                "best_epoch": self._best_epoch,
                "best_dev_metric": self._best_metric,
            }
        )
        return improved

    def _is_improvement(self, value: float, current_best: float) -> bool:
        if self.higher_is_better:
            return value > current_best
        return value < current_best


class _DenseFinetuneLossObserver:
    def __init__(self, loss: Any) -> None:
        self._loss = loss
        self._handle: Any | None = None
        self._weighted_loss = 0.0
        self._weight = 0

    def install(self) -> None:
        register_forward_hook = getattr(self._loss, "register_forward_hook", None)
        if callable(register_forward_hook):
            self._handle = register_forward_hook(self._capture_loss)

    def remove(self) -> None:
        if self._handle is None:
            return
        remove = getattr(self._handle, "remove", None)
        if callable(remove):
            remove()
        self._handle = None

    def average_and_reset(self) -> float | None:
        if self._weight == 0:
            return None
        average = self._weighted_loss / self._weight
        self._weighted_loss = 0.0
        self._weight = 0
        return average

    def _capture_loss(self, _module: Any, inputs: object, output: object) -> None:
        loss = _scalar_loss_value(output)
        batch_size = _infer_batch_size(inputs)
        self._weighted_loss += loss * batch_size
        self._weight += batch_size


def _scalar_loss_value(output: object) -> float:
    value = output
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    mean = getattr(value, "mean", None)
    if callable(mean):
        value = mean()
    item = getattr(value, "item", None)
    if callable(item):
        return float(cast(float, item()))
    return float(cast(float, value))


def _infer_batch_size(inputs: object) -> int:
    features = inputs[0] if isinstance(inputs, tuple) and inputs else inputs
    batch_size = _batch_size_from_features(features)
    return 1 if batch_size is None or batch_size <= 0 else batch_size


def _batch_size_from_features(features: object) -> int | None:
    if isinstance(features, Mapping):
        return _batch_size_from_mapping(features)
    if isinstance(features, Sequence) and not isinstance(features, (str, bytes)):
        for feature in features:
            batch_size = _batch_size_from_features(feature)
            if batch_size is not None:
                return batch_size
    return None


def _batch_size_from_mapping(features: Mapping[object, object]) -> int | None:
    for key in ("input_ids", "attention_mask", "token_type_ids"):
        if key in features:
            batch_size = _tensor_batch_size(features[key])
            if batch_size is not None:
                return batch_size
    for value in features.values():
        batch_size = _tensor_batch_size(value)
        if batch_size is not None:
            return batch_size
    return None


def _tensor_batch_size(value: object) -> int | None:
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[0])
    size = getattr(value, "size", None)
    if callable(size):
        try:
            return int(cast(int, size(0)))
        except TypeError:
            return None
    return None


def _train_loader_len(train_loader: object) -> int:
    try:
        return int(len(cast(Any, train_loader)))
    except TypeError:
        return 0


@dataclass
class _SentenceTransformers27FitRunner:
    model: DenseFinetuneModel
    train_loader: Any
    loss: Any
    evaluator: Any
    config: DenseFinetuneRunConfig
    output_dir: Path
    model_dir: Path
    metric_records: tuple[dict[str, object], ...] = field(init=False, default=())

    def train(self) -> None:
        tracker = _DenseFinetuneMetricTracker(
            selected_metric_name=self.config.selection.best_metric,
            higher_is_better=self.config.selection.higher_is_better,
        )
        baseline_score = float(self.evaluator(self.model, output_path=str(self.output_dir)))
        tracker.record(
            epoch=0,
            global_step=0,
            train_loss=None,
            selected_metric_value=baseline_score,
        )
        self.model_dir.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(self.model_dir))
        loss_observer = _DenseFinetuneLossObserver(self.loss)
        loss_observer.install()
        steps_per_epoch = _train_loader_len(self.train_loader)

        def record_epoch(score: float, epoch: int, steps: int) -> None:
            epoch_number = int(epoch) + 1
            global_step = (
                epoch_number * steps_per_epoch
                if int(steps) < 0
                else int(epoch) * steps_per_epoch + int(steps)
            )
            improved = tracker.record(
                epoch=epoch_number,
                global_step=global_step,
                train_loss=loss_observer.average_and_reset(),
                selected_metric_value=float(score),
            )
            if improved:
                self.model.save(str(self.model_dir))

        try:
            self.model.fit(
                train_objectives=[(self.train_loader, self.loss)],
                evaluator=self.evaluator,
                epochs=self.config.trainer.epochs,
                warmup_steps=self.config.trainer.warmup_steps,
                optimizer_params={"lr": self.config.trainer.learning_rate},
                max_grad_norm=self.config.trainer.max_grad_norm,
                use_amp=self.config.trainer.use_amp,
                output_path=str(self.output_dir),
                save_best_model=False,
                callback=record_epoch,
            )
        finally:
            loss_observer.remove()
        self.metric_records = tracker.records


def train_dense_finetune(
    *,
    config: DenseFinetuneRunConfig,
    train_requests: Sequence[TextRankingRequest],
    train_pairs: Sequence[TrainPairRecord],
    dev_requests: Sequence[TextRankingRequest],
    dev_labels: Sequence[EvidenceLabel],
    output_dir: Path,
    model_dir: Path,
    model_factory: DenseFinetuneModelFactory | None = None,
    trainer_factory: DenseFinetuneTrainerFactory | None = None,
) -> DenseFinetuneTrainingResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    examples = build_dense_finetune_examples(
        ranking_requests=train_requests,
        train_pairs=train_pairs,
        settings=config.data,
        query_prefix=config.query_prefix,
        passage_prefix=config.passage_prefix,
    )
    evaluator_payload = build_ir_evaluator_payload(
        ranking_requests=dev_requests,
        labels=dev_labels,
        query_prefix=config.query_prefix,
        passage_prefix=config.passage_prefix,
    )
    model = (model_factory or _load_sentence_transformer)(config.base_model, config.trainer.device)
    trainer = (trainer_factory or _build_sentence_transformers_fit_runner)(
        DenseFinetuneTrainerRequest(
            model=model,
            train_rows=examples.rows,
            evaluator_payload=evaluator_payload,
            config=config,
            output_dir=output_dir,
            model_dir=model_dir,
        )
    )
    trainer.train()
    metric_records = _trainer_metric_records(trainer)
    metadata_path = write_dense_ft_model_metadata(
        model_dir=model_dir,
        metadata=DenseFinetuneModelMetadata(
            base_model=config.base_model,
            query_prefix=config.query_prefix,
            passage_prefix=config.passage_prefix,
            batch_size=config.batch_size,
            device=config.trainer.device,
            selection=DenseFinetuneSelectionMetadata(
                selected_metric=config.selection.best_metric,
                higher_is_better=config.selection.higher_is_better,
            ),
        ),
    )
    selected_value = _selected_metric_value(metric_records, config.selection.best_metric)
    return DenseFinetuneTrainingResult(
        model_dir=model_dir,
        metadata_path=metadata_path,
        selected_metric_name=config.selection.best_metric,
        selected_metric_value=None if selected_value is None else float(selected_value),
        metric_records=metric_records,
    )


def _trainer_metric_records(trainer: DenseFinetuneTrainer) -> tuple[dict[str, object], ...]:
    records = getattr(trainer, "metric_records", None)
    if not isinstance(records, (list, tuple)):
        raise TypeError(f"Dense-ft trainer has invalid metric_records: {type(records).__name__}.")
    return tuple(dict(record) for record in records)


def _selected_metric_value(
    metric_records: Sequence[Mapping[str, object]],
    selected_metric_name: str,
) -> float | None:
    if not metric_records:
        return None
    last_record = metric_records[-1]
    value = last_record.get("best_dev_metric", last_record.get(selected_metric_name))
    return None if value is None else float(cast(float, value))


def _load_sentence_transformer(model_name: str, device: str) -> DenseFinetuneModel:
    try:
        return cast(DenseFinetuneModel, cast(object, load_sentence_transformer(model_name, device=device)))
    except RuntimeError as error:
        raise RuntimeError("sentence-transformers is required for dense-ft training.") from error


def _build_sentence_transformers_fit_runner(request: DenseFinetuneTrainerRequest) -> DenseFinetuneTrainer:
    components = _load_sentence_transformers_training_components()
    random.seed(request.config.trainer.random_seed)
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("torch is required for dense-ft training.") from error
    torch.manual_seed(request.config.trainer.random_seed)
    train_examples = [
        components.InputExample(texts=[row[key] for key in ("anchor", "positive", "negative") if key in row])
        for row in request.train_rows
    ]
    train_loader = components.DataLoader(
        train_examples,
        shuffle=True,
        batch_size=request.config.trainer.train_batch_size,
    )
    evaluator = components.InformationRetrievalEvaluator(
        queries=request.evaluator_payload.queries,
        corpus=request.evaluator_payload.corpus,
        relevant_docs=request.evaluator_payload.relevant_docs,
        name="dev",
        main_score_function="cos_sim",
        ndcg_at_k=[10],
        accuracy_at_k=[1, 3, 5, 10],
        precision_recall_at_k=[1, 3, 5, 10],
        batch_size=request.config.trainer.eval_batch_size,
        write_csv=False,
    )
    loss = components.MultipleNegativesRankingLoss(request.model)
    return _SentenceTransformers27FitRunner(
        model=request.model,
        train_loader=train_loader,
        loss=loss,
        evaluator=evaluator,
        config=request.config,
        output_dir=request.output_dir,
        model_dir=request.model_dir,
    )


def _load_sentence_transformers_training_components() -> DenseFinetuneTrainingComponents:
    try:
        from importlib import import_module

        from sentence_transformers import InputExample
        from torch.utils.data import DataLoader
        InformationRetrievalEvaluator = import_module(
            "sentence_transformers.evaluation"
        ).InformationRetrievalEvaluator
        MultipleNegativesRankingLoss = import_module(
            "sentence_transformers.losses"
        ).MultipleNegativesRankingLoss
    except ImportError as error:
        raise RuntimeError("sentence-transformers==2.7.0 and torch are required for dense-ft training.") from error
    return DenseFinetuneTrainingComponents(
        InputExample=InputExample,
        DataLoader=DataLoader,
        InformationRetrievalEvaluator=InformationRetrievalEvaluator,
        MultipleNegativesRankingLoss=MultipleNegativesRankingLoss,
    )

from __future__ import annotations

import random
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from graph_memory.contracts.model import (
    DomainModel,
    NonEmptyStr,
    NonNegativeFiniteFloat,
    NonNegativeInt,
    PositiveFiniteFloat,
    PositiveInt,
)
from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.models.cross_encoder.contracts import (
    CrossEncoderExample,
    CrossEncoderTrainingResult,
)
from graph_memory.models.cross_encoder.metadata import (
    CrossEncoderModelMetadata,
    write_cross_encoder_model_metadata,
)
from graph_memory.retrieval.methods.flat.cross_encoder import load_cross_encoder
from graph_memory.retrieval.methods.ids import DenseCandidateView
from graph_memory.retrieval.requests import TextRankingRequest
from graph_memory.training_pairs.contracts import TrainPairRecord


class CrossEncoderTrainerSettings(DomainModel):
    learning_rate: PositiveFiniteFloat = 2e-5
    train_batch_size: PositiveInt = 16
    eval_batch_size: PositiveInt = 64
    epochs: PositiveInt = 1
    warmup_steps: NonNegativeInt = 0
    weight_decay: NonNegativeFiniteFloat = 0.01
    max_grad_norm: NonNegativeFiniteFloat = 1.0
    random_seed: int = 13
    device: NonEmptyStr
    use_amp: bool = False


class CrossEncoderRunConfig(DomainModel):
    variant: DenseCandidateView = "flat"
    base_model: NonEmptyStr
    max_length: PositiveInt = 512
    trainer: CrossEncoderTrainerSettings


class CrossEncoderModelLike:
    def predict(self, sentences: object, **kwargs: object) -> object: ...
    def fit(self, **kwargs: object) -> None: ...
    def save(self, path: str, **kwargs: object) -> None: ...


class _RecordingBceLoss(torch.nn.BCEWithLogitsLoss):
    def __init__(self) -> None:
        super().__init__()
        self._weighted_loss = 0.0
        self._weight = 0

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        value = super().forward(input, target)
        weight = int(target.numel())
        self._weighted_loss += float(value.detach().cpu()) * weight
        self._weight += weight
        return value

    def average_and_reset(self) -> float | None:
        if self._weight == 0:
            return None
        value = self._weighted_loss / self._weight
        self._weighted_loss = 0.0
        self._weight = 0
        return value


@dataclass
class _TaskLocalCrossEncoderEvaluator:
    requests: tuple[TextRankingRequest, ...]
    labels: tuple[EvidenceLabel, ...]
    batch_size: int
    metric_values: dict[str, float] = field(init=False, default_factory=dict)

    def __call__(
        self,
        model: object,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> float:
        del output_path, epoch, steps
        labels_by_id = {label.task_id: label for label in self.labels}
        request_ids = {request.task_id for request in self.requests}
        if set(labels_by_id) != request_ids:
            raise ValueError("Cross-Encoder dev requests and labels must align")
        metric_rows: list[dict[str, float]] = []
        predict = getattr(model, "predict", None)
        if not callable(predict):
            raise TypeError("Cross-Encoder evaluator requires model.predict")
        all_pairs = [
            [request.query_text, candidate.text]
            for request in self.requests
            for candidate in request.candidates
        ]
        raw = predict(
            all_pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        all_scores = np.asarray(raw, dtype=float).reshape(-1)
        if all_scores.shape != (len(all_pairs),):
            raise ValueError(
                "Cross-Encoder dev evaluator received invalid scores: "
                f"expected={(len(all_pairs),)} observed={all_scores.shape}"
            )
        offset = 0
        for request in self.requests:
            scores = all_scores[offset : offset + len(request.candidates)]
            offset += len(request.candidates)
            order = sorted(
                range(len(request.candidates)),
                key=lambda index: (
                    -float(scores[index]),
                    request.candidates[index].item_id,
                ),
            )
            ranked_ids = [request.candidates[index].item_id for index in order]
            gold = set(labels_by_id[request.task_id].gold_evidence_item_ids)
            if not gold:
                raise ValueError(
                    f"Cross-Encoder dev task={request.task_id} has no gold candidates"
                )
            metric_rows.append(
                {
                    f"dev_recall_at_{k}": len(gold.intersection(ranked_ids[:k]))
                    / len(gold)
                    for k in (2, 5, 10)
                }
            )
        if not metric_rows:
            raise ValueError("Cross-Encoder dev evaluation requires at least one task")
        self.metric_values = {
            name: sum(row[name] for row in metric_rows) / len(metric_rows)
            for name in metric_rows[0]
        }
        return self.metric_values["dev_recall_at_5"]


def build_cross_encoder_examples(
    *,
    ranking_requests: Sequence[TextRankingRequest],
    train_pairs: Sequence[TrainPairRecord],
) -> tuple[CrossEncoderExample, ...]:
    request_by_id = {request.task_id: request for request in ranking_requests}
    if len(request_by_id) != len(ranking_requests):
        raise ValueError("Cross-Encoder train request task IDs must be unique")
    candidate_by_task = {
        task_id: {candidate.item_id: candidate for candidate in request.candidates}
        for task_id, request in request_by_id.items()
    }
    examples: list[CrossEncoderExample] = []
    seen: set[tuple[str, str, str]] = set()
    for pair in train_pairs:
        request = request_by_id.get(pair.task_id)
        if request is None:
            raise ValueError(
                f"Cross-Encoder pair task_id={pair.task_id} has no request"
            )
        candidate = candidate_by_task[pair.task_id].get(pair.node_id)
        if candidate is None:
            raise ValueError(
                f"Cross-Encoder pair task_id={pair.task_id} node_id={pair.node_id} "
                "has no candidate"
            )
        key = (pair.task_id, pair.node_id, pair.sample_type)
        if key in seen:
            raise ValueError(f"Duplicate Cross-Encoder pair={key}")
        seen.add(key)
        examples.append(
            CrossEncoderExample(
                task_id=pair.task_id,
                candidate_id=pair.node_id,
                query_text=request.query_text,
                candidate_text=candidate.text,
                label=float(pair.label),
            )
        )
    if not examples:
        raise ValueError("Cross-Encoder training requires at least one pair")
    return tuple(examples)


def train_cross_encoder(
    *,
    config: CrossEncoderRunConfig,
    train_requests: Sequence[TextRankingRequest],
    train_pairs: Sequence[TrainPairRecord],
    dev_requests: Sequence[TextRankingRequest],
    dev_labels: Sequence[EvidenceLabel],
    output_dir: Path,
    model_dir: Path,
) -> CrossEncoderTrainingResult:
    examples = build_cross_encoder_examples(
        ranking_requests=train_requests,
        train_pairs=train_pairs,
    )
    components = _training_components()
    _seed_everything(config.trainer.random_seed)
    model = cast(
        CrossEncoderModelLike,
        load_cross_encoder(
            config.base_model,
            device=config.trainer.device,
            max_length=config.max_length,
            num_labels=1,
        ),
    )
    input_examples = [
        components["InputExample"](
            texts=[example.query_text, example.candidate_text],
            label=example.label,
        )
        for example in examples
    ]
    generator = components["torch"].Generator()
    generator.manual_seed(config.trainer.random_seed)
    train_loader = components["DataLoader"](
        input_examples,
        shuffle=True,
        batch_size=config.trainer.train_batch_size,
        generator=generator,
    )
    evaluator = _TaskLocalCrossEncoderEvaluator(
        requests=tuple(dev_requests),
        labels=tuple(dev_labels),
        batch_size=config.trainer.eval_batch_size,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    best_score = float("-inf")
    best_epoch = 0
    loss = _RecordingBceLoss()
    steps_per_epoch = len(train_loader)

    def record_epoch(score: float, epoch: int, steps: int) -> None:
        nonlocal best_score, best_epoch
        epoch_number = int(epoch) + 1
        if float(score) > best_score:
            best_score = float(score)
            best_epoch = epoch_number
            model.save(str(model_dir), safe_serialization=True)
        global_step = (
            epoch_number * steps_per_epoch
            if int(steps) < 0
            else int(epoch) * steps_per_epoch + int(steps)
        )
        records.append(
            _metric_record(
                epoch=epoch_number,
                global_step=global_step,
                train_loss=loss.average_and_reset(),
                score=float(score),
                best_epoch=best_epoch,
                best_score=best_score,
                values=evaluator.metric_values,
            )
        )

    model.fit(
        train_dataloader=train_loader,
        evaluator=evaluator,
        epochs=config.trainer.epochs,
        loss_fct=loss,
        warmup_steps=config.trainer.warmup_steps,
        optimizer_params={"lr": config.trainer.learning_rate},
        weight_decay=config.trainer.weight_decay,
        max_grad_norm=config.trainer.max_grad_norm,
        use_amp=config.trainer.use_amp,
        output_path=str(output_dir),
        save_best_model=False,
        callback=record_epoch,
        show_progress_bar=True,
    )
    shutil.rmtree(output_dir, ignore_errors=True)
    if best_epoch <= 0 or not np.isfinite(best_score):
        raise RuntimeError("Cross-Encoder training did not produce a selected checkpoint")
    write_cross_encoder_model_metadata(
        model_dir=model_dir,
        metadata=CrossEncoderModelMetadata(
            base_model=config.base_model,
            max_length=config.max_length,
            train_batch_size=config.trainer.train_batch_size,
            eval_batch_size=config.trainer.eval_batch_size,
            device=config.trainer.device,
            variant=config.variant,
        ),
    )
    return CrossEncoderTrainingResult(
        model_dir=model_dir.as_posix(),
        selected_metric_name="dev_recall_at_5",
        selected_metric_value=best_score,
        metric_records=tuple(records),
        variant=config.variant,
    )


def _metric_record(
    *,
    epoch: int,
    global_step: int,
    train_loss: float | None,
    score: float,
    best_epoch: int,
    best_score: float,
    values: Mapping[str, float],
) -> dict[str, object]:
    return {
        "epoch": epoch,
        "global_step": global_step,
        "train_loss": train_loss,
        "dev_recall_at_5": score,
        "best_epoch": best_epoch,
        "best_dev_metric": best_score,
        **dict(values),
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _training_components() -> dict[str, Any]:
    try:
        from sentence_transformers import InputExample
        from torch.utils.data import DataLoader
    except ImportError as error:
        raise RuntimeError(
            "sentence-transformers==2.7.0 and torch are required for Cross-Encoder training."
        ) from error
    return {"torch": torch, "InputExample": InputExample, "DataLoader": DataLoader}


__all__ = [
    "CrossEncoderRunConfig",
    "CrossEncoderTrainerSettings",
    "build_cross_encoder_examples",
    "train_cross_encoder",
]

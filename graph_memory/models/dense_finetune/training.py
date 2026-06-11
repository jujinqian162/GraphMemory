from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from graph_memory.contracts.tasks import MemoryTaskInput, MemoryTaskLabels
from graph_memory.contracts.training_pairs import TrainPairRecord
from graph_memory.models.dense_finetune.contracts import DenseFinetuneDataSettings, DenseFinetuneIREvaluatorPayload
from graph_memory.models.dense_finetune.data import build_dense_finetune_examples, build_ir_evaluator_payload

DENSE_FT_METADATA_FILENAME = "dense_ft_model_config.json"


@dataclass(frozen=True)
class DenseFinetuneTrainerSettings:
    learning_rate: float = 2e-5
    train_batch_size: int = 16
    eval_batch_size: int = 64
    epochs: int = 1
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    random_seed: int = 13
    device: str = "cuda"
    fp16: bool = False
    bf16: bool = False
    logging_steps: int = 50
    save_total_limit: int = 2


@dataclass(frozen=True)
class DenseFinetuneSelectionSettings:
    best_metric: str = "eval_dev_cosine_ndcg@10"
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
    def save(self, output_path: str | Path) -> None:
        ...


class DenseFinetuneTrainer(Protocol):
    def train(self) -> None:
        ...

    def evaluate(self) -> Mapping[str, float]:
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


DenseFinetuneModelFactory = Callable[[str], DenseFinetuneModel]
DenseFinetuneTrainerFactory = Callable[[DenseFinetuneTrainerRequest], DenseFinetuneTrainer]


@dataclass(frozen=True)
class DenseFinetuneTrainingComponents:
    Dataset: Any
    SentenceTransformerTrainer: Any
    SentenceTransformerTrainingArguments: Any
    InformationRetrievalEvaluator: Any
    MultipleNegativesRankingLoss: Any


def train_dense_finetune(
    *,
    config: DenseFinetuneRunConfig,
    train_task_inputs: Sequence[MemoryTaskInput],
    train_pairs: Sequence[TrainPairRecord],
    dev_task_inputs: Sequence[MemoryTaskInput],
    dev_labels: Sequence[MemoryTaskLabels],
    output_dir: Path,
    model_dir: Path,
    model_factory: DenseFinetuneModelFactory | None = None,
    trainer_factory: DenseFinetuneTrainerFactory | None = None,
) -> DenseFinetuneTrainingResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    examples = build_dense_finetune_examples(
        task_inputs=train_task_inputs,
        train_pairs=train_pairs,
        settings=config.data,
        query_prefix=config.query_prefix,
        passage_prefix=config.passage_prefix,
    )
    evaluator_payload = build_ir_evaluator_payload(
        task_inputs=dev_task_inputs,
        task_labels=dev_labels,
        query_prefix=config.query_prefix,
        passage_prefix=config.passage_prefix,
    )
    model = (model_factory or _load_sentence_transformer)(config.base_model)
    trainer = (trainer_factory or _build_sentence_transformers_trainer)(
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
    metrics = dict(trainer.evaluate())
    model.save(model_dir)
    metadata_path = write_dense_ft_model_metadata(config=config, model_dir=model_dir)
    selected_value = metrics.get(config.selection.best_metric)
    return DenseFinetuneTrainingResult(
        model_dir=model_dir,
        metadata_path=metadata_path,
        selected_metric_name=config.selection.best_metric,
        selected_metric_value=None if selected_value is None else float(selected_value),
        metric_records=(
            {
                "phase": "final",
                "train_example_count": len(examples.examples),
                "dev_query_count": len(evaluator_payload.queries),
                **metrics,
            },
        ),
    )


def write_dense_ft_model_metadata(*, config: DenseFinetuneRunConfig, model_dir: Path) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "method": "dense_ft",
        "base_model": config.base_model,
        "query_prefix": config.query_prefix,
        "passage_prefix": config.passage_prefix,
        "batch_size": config.batch_size,
        "selection": {
            "selected_metric": config.selection.best_metric,
            "higher_is_better": config.selection.higher_is_better,
        },
    }
    metadata_path = model_dir / DENSE_FT_METADATA_FILENAME
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata_path


def _load_sentence_transformer(model_name: str) -> DenseFinetuneModel:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError("sentence-transformers is required for dense-ft training.") from error
    return cast(DenseFinetuneModel, cast(object, SentenceTransformer(model_name)))


def _build_sentence_transformers_trainer(request: DenseFinetuneTrainerRequest) -> DenseFinetuneTrainer:
    components = _load_sentence_transformers_training_components()
    train_dataset = components.Dataset.from_list(list(request.train_rows))
    evaluator = components.InformationRetrievalEvaluator(
        queries=request.evaluator_payload.queries,
        corpus=request.evaluator_payload.corpus,
        relevant_docs=request.evaluator_payload.relevant_docs,
        name="dev",
        main_score_function="cosine",
        ndcg_at_k=[10],
        accuracy_at_k=[1, 3, 5, 10],
        precision_recall_at_k=[1, 3, 5, 10],
        batch_size=request.config.trainer.eval_batch_size,
        write_csv=False,
    )
    training_args = components.SentenceTransformerTrainingArguments(
        output_dir=str(request.output_dir),
        per_device_train_batch_size=request.config.trainer.train_batch_size,
        per_device_eval_batch_size=request.config.trainer.eval_batch_size,
        num_train_epochs=request.config.trainer.epochs,
        learning_rate=request.config.trainer.learning_rate,
        warmup_ratio=request.config.trainer.warmup_ratio,
        max_grad_norm=request.config.trainer.max_grad_norm,
        logging_steps=request.config.trainer.logging_steps,
        save_total_limit=request.config.trainer.save_total_limit,
        fp16=request.config.trainer.fp16,
        bf16=request.config.trainer.bf16,
        use_cpu=request.config.trainer.device == "cpu",
        seed=request.config.trainer.random_seed,
        report_to="none",
    )
    loss = components.MultipleNegativesRankingLoss(request.model)
    return cast(
        DenseFinetuneTrainer,
        components.SentenceTransformerTrainer(
            model=request.model,
            args=training_args,
            train_dataset=train_dataset,
            loss=loss,
            evaluator=evaluator,
        ),
    )


def _load_sentence_transformers_training_components() -> DenseFinetuneTrainingComponents:
    try:
        from importlib import import_module

        from datasets import Dataset
        from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
        InformationRetrievalEvaluator = import_module(
            "sentence_transformers.evaluation"
        ).InformationRetrievalEvaluator
        MultipleNegativesRankingLoss = import_module(
            "sentence_transformers.losses"
        ).MultipleNegativesRankingLoss
    except ImportError as error:
        raise RuntimeError(
            "datasets, accelerate, and sentence-transformers are required for dense-ft training."
        ) from error
    return DenseFinetuneTrainingComponents(
        Dataset=Dataset,
        SentenceTransformerTrainer=SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments=SentenceTransformerTrainingArguments,
        InformationRetrievalEvaluator=InformationRetrievalEvaluator,
        MultipleNegativesRankingLoss=MultipleNegativesRankingLoss,
    )

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import scripts.train_method as train_method_script
from graph_memory.contracts.tasks import MemoryTaskInput, MemoryTaskLabels
from graph_memory.contracts.training_pairs import TrainPairRecord
from graph_memory.models.dense_finetune.training import (
    DenseFinetuneRunConfig,
    DenseFinetuneSelectionSettings,
    DenseFinetuneTrainingComponents,
    DenseFinetuneTrainerRequest,
    DenseFinetuneTrainerSettings,
    _build_sentence_transformers_trainer,
    train_dense_finetune,
)
import graph_memory.models.dense_finetune.training as dense_ft_training
from graph_memory.models.dense_finetune.data import DenseFinetuneDataSettings
from scripts.train_method import main as train_method_main


def _task(task_id: str, *, query: str) -> MemoryTaskInput:
    return {
        "task_id": task_id,
        "query": query,
        "memory_items": [
            {
                "id": "m0",
                "node_type": "document_sentence",
                "source": "Gold",
                "text": "Gold evidence sentence.",
                "sentence_id": 0,
                "position": 0,
            },
            {
                "id": "m1",
                "node_type": "document_sentence",
                "source": "Hard",
                "text": "Hard negative sentence.",
                "sentence_id": 1,
                "position": 1,
            },
        ],
    }


def _labels(task_id: str) -> MemoryTaskLabels:
    return {
        "task_id": task_id,
        "gold_answer": "answer",
        "gold_evidence_nodes": ["m0"],
        "gold_dependency_edges": [],
    }


def _pairs(task_id: str) -> list[TrainPairRecord]:
    return [
        {"task_id": task_id, "node_id": "m0", "label": 1, "sample_type": "positive"},
        {"task_id": task_id, "node_id": "m1", "label": 0, "sample_type": "hard_dense"},
    ]


class FakeSentenceTransformer:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.saved_to: Path | None = None

    def save(self, output_path: str | Path) -> None:
        self.saved_to = Path(output_path)
        self.saved_to.mkdir(parents=True, exist_ok=True)
        (self.saved_to / "modules.json").write_text("[]", encoding="utf-8")


class FakeTrainer:
    def __init__(self, request: DenseFinetuneTrainerRequest) -> None:
        self.request = request
        self.train_called = False

    def train(self) -> None:
        self.train_called = True

    def evaluate(self) -> dict[str, float]:
        return {"eval_dev_cosine_ndcg@10": 0.75}


def test_train_dense_finetune_uses_fake_trainer_and_writes_metadata(tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def model_factory(model_name: str) -> FakeSentenceTransformer:
        model = FakeSentenceTransformer(model_name)
        captured["model"] = model
        return model

    def trainer_factory(request: DenseFinetuneTrainerRequest) -> FakeTrainer:
        captured["request"] = request
        trainer = FakeTrainer(request)
        captured["trainer"] = trainer
        return trainer

    config = DenseFinetuneRunConfig(
        base_model="fake-e5",
        query_prefix="Q: ",
        passage_prefix="P: ",
        batch_size=32,
        data=DenseFinetuneDataSettings(hard_negatives_per_positive=1),
        trainer=DenseFinetuneTrainerSettings(
            learning_rate=2e-5,
            train_batch_size=2,
            eval_batch_size=4,
            epochs=1,
            device="cpu",
            logging_steps=1,
        ),
        selection=DenseFinetuneSelectionSettings(best_metric="eval_dev_cosine_ndcg@10"),
    )
    train_task = _task("train", query="train query")
    dev_task = _task("dev", query="dev query")

    result = train_dense_finetune(
        config=config,
        train_task_inputs=[train_task],
        train_pairs=_pairs("train"),
        dev_task_inputs=[dev_task],
        dev_labels=[_labels("dev")],
        output_dir=tmp_path / "output",
        model_dir=tmp_path / "model",
        model_factory=model_factory,
        trainer_factory=trainer_factory,
    )

    request = captured["request"]
    assert request.train_rows == (
        {
            "anchor": "Q: train query",
            "positive": "P: Gold. Gold evidence sentence.",
            "negative": "P: Hard. Hard negative sentence.",
        },
    )
    assert request.evaluator_payload.queries == {"dev": "Q: dev query"}
    assert request.evaluator_payload.relevant_docs == {"dev": {"dev::m0"}}
    assert captured["trainer"].train_called is True
    assert captured["model"].saved_to == tmp_path / "model"

    metadata = json.loads((tmp_path / "model" / "dense_ft_model_config.json").read_text(encoding="utf-8"))
    assert metadata == {
        "schema_version": 1,
        "method": "dense_ft",
        "base_model": "fake-e5",
        "query_prefix": "Q: ",
        "passage_prefix": "P: ",
        "batch_size": 32,
        "selection": {
            "selected_metric": "eval_dev_cosine_ndcg@10",
            "higher_is_better": True,
        },
    }
    assert result.model_dir == tmp_path / "model"
    assert result.metadata_path == tmp_path / "model" / "dense_ft_model_config.json"
    assert result.selected_metric_name == "eval_dev_cosine_ndcg@10"
    assert result.selected_metric_value == 0.75
    assert result.metric_records == (
        {
            "phase": "final",
            "train_example_count": 1,
            "dev_query_count": 1,
            "eval_dev_cosine_ndcg@10": 0.75,
        },
    )


def test_sentence_transformers_trainer_factory_builds_expected_components(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    class FakeDataset:
        @staticmethod
        def from_list(rows: list[dict[str, str]]) -> object:
            captured["dataset_rows"] = rows
            return {"dataset": rows}

    class FakeTrainingArguments:
        def __init__(self, **kwargs: object) -> None:
            captured["training_args"] = kwargs

    class FakeLoss:
        def __init__(self, model: object) -> None:
            captured["loss_model"] = model

    class FakeEvaluator:
        def __init__(self, **kwargs: object) -> None:
            captured["evaluator"] = kwargs

    class FakeTrainer:
        def __init__(self, **kwargs: object) -> None:
            captured["trainer"] = kwargs

        def train(self) -> None:
            pass

        def evaluate(self) -> dict[str, float]:
            return {}

    monkeypatch.setattr(
        dense_ft_training,
        "_load_sentence_transformers_training_components",
        lambda: DenseFinetuneTrainingComponents(
            Dataset=FakeDataset,
            SentenceTransformerTrainer=FakeTrainer,
            SentenceTransformerTrainingArguments=FakeTrainingArguments,
            InformationRetrievalEvaluator=FakeEvaluator,
            MultipleNegativesRankingLoss=FakeLoss,
        ),
    )
    model = FakeSentenceTransformer("fake-e5")
    config = DenseFinetuneRunConfig(
        base_model="fake-e5",
        query_prefix="Q: ",
        passage_prefix="P: ",
        batch_size=48,
        data=DenseFinetuneDataSettings(),
        trainer=DenseFinetuneTrainerSettings(
            learning_rate=0.123,
            train_batch_size=3,
            eval_batch_size=5,
            epochs=7,
            warmup_ratio=0.2,
            max_grad_norm=0.9,
            random_seed=99,
            device="cpu",
            fp16=True,
            bf16=False,
            logging_steps=11,
            save_total_limit=4,
        ),
        selection=DenseFinetuneSelectionSettings(best_metric="eval_dev_cosine_ndcg@10"),
    )

    trainer = _build_sentence_transformers_trainer(
        DenseFinetuneTrainerRequest(
            model=model,
            train_rows=(
                {"anchor": "Q: train", "positive": "P: positive", "negative": "P: negative"},
            ),
            evaluator_payload=dense_ft_training.DenseFinetuneIREvaluatorPayload(
                queries={"dev": "Q: dev"},
                corpus={"dev::m0": "P: positive"},
                relevant_docs={"dev": {"dev::m0"}},
            ),
            config=config,
            output_dir=tmp_path / "output",
            model_dir=tmp_path / "model",
        )
    )

    assert isinstance(trainer, FakeTrainer)
    assert captured["dataset_rows"] == [{"anchor": "Q: train", "positive": "P: positive", "negative": "P: negative"}]
    assert captured["loss_model"] is model
    assert captured["evaluator"] == {
        "queries": {"dev": "Q: dev"},
        "corpus": {"dev::m0": "P: positive"},
        "relevant_docs": {"dev": {"dev::m0"}},
        "name": "dev",
        "main_score_function": "cosine",
        "ndcg_at_k": [10],
        "accuracy_at_k": [1, 3, 5, 10],
        "precision_recall_at_k": [1, 3, 5, 10],
        "batch_size": 5,
        "write_csv": False,
    }
    assert captured["training_args"] == {
        "output_dir": str(tmp_path / "output"),
        "per_device_train_batch_size": 3,
        "per_device_eval_batch_size": 5,
        "num_train_epochs": 7,
        "learning_rate": 0.123,
        "warmup_ratio": 0.2,
        "max_grad_norm": 0.9,
        "logging_steps": 11,
        "save_total_limit": 4,
        "fp16": True,
        "bf16": False,
        "use_cpu": True,
        "seed": 99,
        "report_to": "none",
    }
    assert captured["trainer"]["model"] is model
    assert captured["trainer"]["train_dataset"] == {"dataset": captured["dataset_rows"]}
    assert isinstance(captured["trainer"]["args"], FakeTrainingArguments)
    assert isinstance(captured["trainer"]["loss"], FakeLoss)
    assert isinstance(captured["trainer"]["evaluator"], FakeEvaluator)


def test_dense_ft_train_method_cli_writes_model_metrics_and_summary(monkeypatch, tmp_path: Path) -> None:
    train_tasks_path = tmp_path / "train.input.json"
    train_labels_path = tmp_path / "train.labels.json"
    train_pairs_path = tmp_path / "train.pairs.json"
    dev_tasks_path = tmp_path / "dev.input.json"
    dev_labels_path = tmp_path / "dev.labels.json"
    output_dir = tmp_path / "dense_ft_run"
    model_dir = output_dir / "checkpoints" / "best_model"
    train_tasks_path.write_text(json.dumps([_task("train", query="train query")]), encoding="utf-8")
    train_labels_path.write_text(json.dumps([_labels("train")]), encoding="utf-8")
    train_pairs_path.write_text(json.dumps(_pairs("train")), encoding="utf-8")
    dev_tasks_path.write_text(json.dumps([_task("dev", query="dev query")]), encoding="utf-8")
    dev_labels_path.write_text(json.dumps([_labels("dev")]), encoding="utf-8")
    monkeypatch.setattr(
        dense_ft_training,
        "_load_sentence_transformer",
        lambda model_name: FakeSentenceTransformer(model_name),
    )
    monkeypatch.setattr(
        dense_ft_training,
        "_build_sentence_transformers_trainer",
        lambda request: FakeTrainer(request),
    )

    exit_code = train_method_main(
        [
            "--method",
            "dense_ft",
            "--train_tasks",
            str(train_tasks_path),
            "--train_labels",
            str(train_labels_path),
            "--train_pairs",
            str(train_pairs_path),
            "--dev_tasks",
            str(dev_tasks_path),
            "--dev_labels",
            str(dev_labels_path),
            "--output_dir",
            str(output_dir),
            "--model_dir",
            str(model_dir),
            "--encoder_model",
            "fake-e5",
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0
    assert (model_dir / "modules.json").exists()
    assert (model_dir / "dense_ft_model_config.json").exists()
    assert (output_dir / "train_metrics.jsonl").exists()
    summary = json.loads((output_dir / "train_run_summary.json").read_text(encoding="utf-8"))
    assert summary["script"] == "train_method.py"
    assert summary["effective_config"]["method"] == "dense_ft"
    assert summary["outputs"]["best_checkpoint"] == str(model_dir)
    assert "train_graphs" not in summary["inputs"]
    assert "dev_graphs" not in summary["inputs"]
    assert "train_graph_retriever" not in inspect.getsource(train_method_script)

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, cast

import pytest
import scripts.train_method as train_method_script
from graph_memory.config import CONFIG_LOADER
from graph_memory.datasets.hotpotqa.projectors import HotpotQAToTextRankingRequest
from graph_memory.datasets.hotpotqa.records import HotpotQARankingRecord, HotpotQALabelRecord
from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.contracts.training_pairs import TrainPairRecord
from graph_memory.io import write_json
from graph_memory.models.dense_finetune.training import (
    DenseFinetuneRunConfig,
    DenseFinetuneSelectionSettings,
    DenseFinetuneTrainingComponents,
    DenseFinetuneTrainerRequest,
    DenseFinetuneTrainerSettings,
    _build_sentence_transformers_fit_runner,
    train_dense_finetune,
)
import graph_memory.models.dense_finetune.training as dense_ft_training
from graph_memory.models.dense_finetune.data import DenseFinetuneDataSettings
from graph_memory.registry.method_configs import DenseFinetuneMethodSettings
from graph_memory.registry.retrieval import DenseEncoderSettings, RetrievalMethodId
from graph_memory.registry.stage_configs import DenseFinetuneTrainIO, DenseFinetuneTrainStageConfig
from scripts.train_method import main as train_method_main


def _task(task_id: str, *, query: str) -> HotpotQARankingRecord:
    return {
        "task_id": task_id,
        "question": query,
        "candidate_sentences": [
            {
                "sentence_id": "m0",
                "title": "Gold",
                "text": "Gold evidence sentence.",
                "sentence_index": 0,
                "position": 0,
            },
            {
                "sentence_id": "m1",
                "title": "Hard",
                "text": "Hard negative sentence.",
                "sentence_index": 1,
                "position": 1,
            },
        ],
    }


def _labels(task_id: str) -> HotpotQALabelRecord:
    return {
        "task_id": task_id,
        "gold_answer": "answer",
        "gold_evidence_sentence_ids": ["m0"],
        "gold_dependency_edges": [],
    }


def _request(task: HotpotQARankingRecord):
    return HotpotQAToTextRankingRequest().project(task)


def _evidence_label(label: HotpotQALabelRecord) -> EvidenceLabel:
    return EvidenceLabel(
        task_id=label["task_id"],
        gold_answer=label["gold_answer"],
        gold_evidence_item_ids=tuple(label["gold_evidence_sentence_ids"]),
        gold_dependency_edges=tuple((edge[0], edge[1]) for edge in label["gold_dependency_edges"]),
    )


def _pairs(task_id: str) -> list[TrainPairRecord]:
    return [
        {"task_id": task_id, "node_id": "m0", "label": 1, "sample_type": "positive"},
        {"task_id": task_id, "node_id": "m1", "label": 0, "sample_type": "hard_dense"},
    ]


def write_dense_ft_train_stage_config(
    path: Path,
    *,
    train_tasks_path: Path,
    train_labels_path: Path,
    train_pairs_path: Path,
    dev_tasks_path: Path,
    dev_labels_path: Path,
    output_dir: Path,
    model_dir: Path,
) -> None:
    config = DenseFinetuneTrainStageConfig(
        method=RetrievalMethodId.DENSE_FT,
        io=DenseFinetuneTrainIO(
            train_tasks=train_tasks_path,
            train_labels=train_labels_path,
            train_pairs=train_pairs_path,
            dev_tasks=dev_tasks_path,
            dev_labels=dev_labels_path,
            output_dir=output_dir,
            model_dir=model_dir,
            metrics=output_dir / "train_metrics.jsonl",
            run_summary=output_dir / "train_run_summary.json",
        ),
        job=DenseFinetuneMethodSettings(
            encoder=DenseEncoderSettings(
                model_name="fake-e5",
                query_prefix="query: ",
                passage_prefix="passage: ",
                batch_size=64,
            ),
            trainer=DenseFinetuneTrainerSettings(device="cpu"),
        ),
    )
    write_json(path, CONFIG_LOADER.to_json(config))


class FakeSentenceTransformer:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.saved_to: Path | None = None
        self.save_events: list[Path] = []
        self.save_marker = "baseline"
        self.fit_kwargs: dict[str, object] | None = None

    def fit(self, **kwargs: object) -> None:
        self.fit_kwargs = kwargs

    def save(self, output_path: str) -> None:
        if not isinstance(output_path, str):
            raise TypeError("SentenceTransformers 2.7.0 save() requires a string path")
        self.saved_to = Path(output_path)
        self.save_events.append(self.saved_to)
        self.saved_to.mkdir(parents=True, exist_ok=True)
        (self.saved_to / "modules.json").write_text("[]", encoding="utf-8")
        (self.saved_to / "state.txt").write_text(self.save_marker, encoding="utf-8")


class CallbackSentenceTransformer(FakeSentenceTransformer):
    def __init__(
        self,
        model_name: str,
        *,
        epoch_scores: tuple[float, ...],
        epoch_losses: tuple[tuple[tuple[float, int], ...], ...] = (),
        fail_fit: bool = False,
    ) -> None:
        super().__init__(model_name)
        self.epoch_scores = epoch_scores
        self.epoch_losses = epoch_losses
        self.fail_fit = fail_fit

    def fit(self, **kwargs: object) -> None:
        super().fit(**kwargs)
        train_objectives = cast(list[tuple[object, object]], kwargs["train_objectives"])
        loss = train_objectives[0][1]
        callback = cast(Any, kwargs["callback"])
        if self.fail_fit:
            raise RuntimeError("fit failed")
        for epoch_index, score in enumerate(self.epoch_scores, start=1):
            self.save_marker = f"epoch-{epoch_index}"
            if hasattr(loss, "emit_loss") and epoch_index <= len(self.epoch_losses):
                for loss_value, batch_size in self.epoch_losses[epoch_index - 1]:
                    cast(Any, loss).emit_loss(loss_value, batch_size)
            callback(score, epoch_index - 1, -1)


class _FakeHookHandle:
    def __init__(self) -> None:
        self.removed = False

    def remove(self) -> None:
        self.removed = True


class _FakeLossScalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def detach(self) -> _FakeLossScalar:
        return self

    def mean(self) -> _FakeLossScalar:
        return self

    def item(self) -> float:
        return self.value


class _FakeBatchTensor:
    def __init__(self, batch_size: int) -> None:
        self.shape = (batch_size, 3)

    def size(self, index: int) -> int:
        return self.shape[index]


class _HookableFakeLoss:
    def __init__(self, model: object) -> None:
        self.model = model
        self.hook: Any | None = None
        self.handle = _FakeHookHandle()

    def register_forward_hook(self, hook: Any) -> _FakeHookHandle:
        self.hook = hook
        return self.handle

    def emit_loss(self, value: float, batch_size: int) -> None:
        if self.hook is None:
            return
        features = [{"input_ids": _FakeBatchTensor(batch_size)}]
        self.hook(self, (features, object()), _FakeLossScalar(value))


class FakeTrainer:
    def __init__(self, request: DenseFinetuneTrainerRequest) -> None:
        self.request = request
        self.train_called = False
        self.metric_records = (
            {
                "epoch": 0,
                "global_step": 0,
                "train_loss": None,
                "eval_dev_cos_sim_map@100": 0.7,
                "best_epoch": 0,
                "best_dev_metric": 0.7,
            },
            {
                "epoch": 1,
                "global_step": 1,
                "train_loss": 0.25,
                "eval_dev_cos_sim_map@100": 0.75,
                "best_epoch": 1,
                "best_dev_metric": 0.75,
            },
        )

    def train(self) -> None:
        self.train_called = True
        self.request.model.save(str(self.request.model_dir))


def test_train_dense_finetune_returns_epoch_metric_records_and_writes_metadata(tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def model_factory(model_name: str, device: str) -> FakeSentenceTransformer:
        captured["device"] = device
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
        ),
        selection=DenseFinetuneSelectionSettings(best_metric="eval_dev_cos_sim_map@100"),
    )
    train_task = _task("train", query="train query")
    dev_task = _task("dev", query="dev query")

    result = train_dense_finetune(
        config=config,
        train_requests=[_request(train_task)],
        train_pairs=_pairs("train"),
        dev_requests=[_request(dev_task)],
        dev_labels=[_evidence_label(_labels("dev"))],
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
    assert captured["device"] == "cpu"
    assert captured["model"].saved_to == tmp_path / "model"

    metadata = json.loads((tmp_path / "model" / "dense_ft_model_config.json").read_text(encoding="utf-8"))
    assert metadata == {
        "method": "dense_ft",
        "base_model": "fake-e5",
        "query_prefix": "Q: ",
        "passage_prefix": "P: ",
        "batch_size": 32,
        "device": "cpu",
        "selection": {
            "selected_metric": "eval_dev_cos_sim_map@100",
            "higher_is_better": True,
        },
    }
    assert result.model_dir == tmp_path / "model"
    assert result.metadata_path == tmp_path / "model" / "dense_ft_model_config.json"
    assert result.selected_metric_name == "eval_dev_cos_sim_map@100"
    assert result.selected_metric_value == 0.75
    assert result.metric_records == (
        {
            "epoch": 0,
            "global_step": 0,
            "train_loss": None,
            "eval_dev_cos_sim_map@100": 0.7,
            "best_epoch": 0,
            "best_dev_metric": 0.7,
        },
        {
            "epoch": 1,
            "global_step": 1,
            "train_loss": 0.25,
            "eval_dev_cos_sim_map@100": 0.75,
            "best_epoch": 1,
            "best_dev_metric": 0.75,
        },
    )


def test_sentence_transformers_27_trainer_builds_input_examples_and_calls_fit(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    class FakeInputExample:
        def __init__(self, *, texts: list[str]) -> None:
            self.texts = texts

    class FakeDataLoader:
        def __init__(self, dataset: list[FakeInputExample], *, shuffle: bool, batch_size: int) -> None:
            self.dataset = dataset
            self.shuffle = shuffle
            self.batch_size = batch_size
            captured["data_loader"] = self

    class FakeLoss:
        def __init__(self, model: object) -> None:
            captured["loss_model"] = model

    class FakeEvaluator:
        def __init__(self, **kwargs: object) -> None:
            captured["evaluator"] = kwargs
            captured["evaluator_instance"] = self

        def __call__(self, model: object, **kwargs: object) -> float:
            captured["evaluation_call"] = {"model": model, **kwargs}
            return 0.75

    monkeypatch.setattr(
        dense_ft_training,
        "_load_sentence_transformers_training_components",
        lambda: DenseFinetuneTrainingComponents(
            InputExample=FakeInputExample,
            DataLoader=FakeDataLoader,
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
            warmup_steps=12,
            max_grad_norm=0.9,
            random_seed=99,
            device="cpu",
            use_amp=True,
        ),
        selection=DenseFinetuneSelectionSettings(best_metric="eval_dev_cos_sim_map@100"),
    )

    trainer = _build_sentence_transformers_fit_runner(
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

    trainer.train()

    data_loader = captured["data_loader"]
    assert [example.texts for example in data_loader.dataset] == [
        ["Q: train", "P: positive", "P: negative"]
    ]
    assert data_loader.shuffle is True
    assert data_loader.batch_size == 3
    assert captured["loss_model"] is model
    assert captured["evaluator"] == {
        "queries": {"dev": "Q: dev"},
        "corpus": {"dev::m0": "P: positive"},
        "relevant_docs": {"dev": {"dev::m0"}},
        "name": "dev",
        "main_score_function": "cos_sim",
        "ndcg_at_k": [10],
        "accuracy_at_k": [1, 3, 5, 10],
        "precision_recall_at_k": [1, 3, 5, 10],
        "batch_size": 5,
        "write_csv": False,
    }
    assert model.fit_kwargs is not None
    fit_kwargs = model.fit_kwargs
    train_objectives = cast(list[tuple[object, object]], fit_kwargs["train_objectives"])
    assert train_objectives == [(data_loader, train_objectives[0][1])]
    assert fit_kwargs["evaluator"] is captured["evaluator_instance"]
    assert fit_kwargs["epochs"] == 7
    assert fit_kwargs["warmup_steps"] == 12
    assert fit_kwargs["optimizer_params"] == {"lr": 0.123}
    assert fit_kwargs["max_grad_norm"] == 0.9
    assert fit_kwargs["use_amp"] is True
    assert fit_kwargs["output_path"] == str(tmp_path / "output")
    assert fit_kwargs["save_best_model"] is False
    assert callable(fit_kwargs["callback"])
    assert captured["evaluation_call"] == {
        "model": model,
        "output_path": str(tmp_path / "output"),
    }
    assert model.saved_to == tmp_path / "model"
    assert trainer.metric_records == (
        {
            "epoch": 0,
            "global_step": 0,
            "train_loss": None,
            "eval_dev_cos_sim_map@100": 0.75,
            "best_epoch": 0,
            "best_dev_metric": 0.75,
        },
    )


def _build_callback_trainer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    model: CallbackSentenceTransformer,
    baseline_score: float,
    higher_is_better: bool = True,
    data_loader_len: int = 2,
) -> tuple[Any, dict[str, Any]]:
    captured: dict[str, Any] = {}

    class FakeInputExample:
        def __init__(self, *, texts: list[str]) -> None:
            self.texts = texts

    class FakeDataLoader:
        def __init__(self, dataset: list[FakeInputExample], *, shuffle: bool, batch_size: int) -> None:
            self.dataset = dataset
            self.shuffle = shuffle
            self.batch_size = batch_size

        def __len__(self) -> int:
            return data_loader_len

    class FakeLoss(_HookableFakeLoss):
        def __init__(self, model: object) -> None:
            super().__init__(model)
            captured["loss"] = self

    class FakeEvaluator:
        def __init__(self, **kwargs: object) -> None:
            captured["evaluator"] = kwargs

        def __call__(self, model: object, **kwargs: object) -> float:
            captured.setdefault("evaluation_calls", []).append({"model": model, **kwargs})
            return baseline_score

    monkeypatch.setattr(
        dense_ft_training,
        "_load_sentence_transformers_training_components",
        lambda: DenseFinetuneTrainingComponents(
            InputExample=FakeInputExample,
            DataLoader=FakeDataLoader,
            InformationRetrievalEvaluator=FakeEvaluator,
            MultipleNegativesRankingLoss=FakeLoss,
        ),
    )

    config = DenseFinetuneRunConfig(
        base_model="fake-e5",
        trainer=DenseFinetuneTrainerSettings(
            train_batch_size=2,
            eval_batch_size=5,
            epochs=max(1, len(model.epoch_scores)),
            device="cpu",
        ),
        selection=DenseFinetuneSelectionSettings(
            best_metric="eval_dev_cos_sim_map@100",
            higher_is_better=higher_is_better,
        ),
    )
    trainer = _build_sentence_transformers_fit_runner(
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
    return trainer, captured


def test_sentence_transformers_27_trainer_averages_epoch_loss_from_forward_hooks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = CallbackSentenceTransformer(
        "fake-e5",
        epoch_scores=(0.8,),
        epoch_losses=(((1.0, 2), (2.0, 4)),),
    )
    trainer, captured = _build_callback_trainer(
        monkeypatch,
        tmp_path,
        model=model,
        baseline_score=0.5,
        data_loader_len=3,
    )

    trainer.train()

    assert trainer.metric_records[1] == {
        "epoch": 1,
        "global_step": 3,
        "train_loss": pytest.approx(10.0 / 6.0),
        "eval_dev_cos_sim_map@100": 0.8,
        "best_epoch": 1,
        "best_dev_metric": 0.8,
    }
    assert captured["loss"].handle.removed is True


def test_sentence_transformers_27_trainer_removes_loss_hook_when_fit_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = CallbackSentenceTransformer("fake-e5", epoch_scores=(), fail_fit=True)
    trainer, captured = _build_callback_trainer(
        monkeypatch,
        tmp_path,
        model=model,
        baseline_score=0.5,
    )

    with pytest.raises(RuntimeError, match="fit failed"):
        trainer.train()

    assert captured["loss"].handle.removed is True


def test_sentence_transformers_27_trainer_keeps_baseline_best_when_epoch_metrics_degrade(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = CallbackSentenceTransformer("fake-e5", epoch_scores=(0.8, 0.7))
    trainer, _captured = _build_callback_trainer(
        monkeypatch,
        tmp_path,
        model=model,
        baseline_score=0.9,
        data_loader_len=4,
    )

    trainer.train()

    assert (tmp_path / "model" / "state.txt").read_text(encoding="utf-8") == "baseline"
    assert model.save_events == [tmp_path / "model"]
    assert trainer.metric_records[-1]["best_epoch"] == 0
    assert trainer.metric_records[-1]["best_dev_metric"] == 0.9


def test_sentence_transformers_27_trainer_overwrites_best_model_when_epoch_improves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = CallbackSentenceTransformer("fake-e5", epoch_scores=(0.6, 0.7))
    trainer, _captured = _build_callback_trainer(
        monkeypatch,
        tmp_path,
        model=model,
        baseline_score=0.5,
        data_loader_len=4,
    )

    trainer.train()

    assert (tmp_path / "model" / "state.txt").read_text(encoding="utf-8") == "epoch-2"
    assert model.save_events == [tmp_path / "model", tmp_path / "model", tmp_path / "model"]
    assert trainer.metric_records[-1]["best_epoch"] == 2
    assert trainer.metric_records[-1]["best_dev_metric"] == 0.7


def test_sentence_transformers_27_trainer_respects_lower_is_better_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = CallbackSentenceTransformer("fake-e5", epoch_scores=(0.4, 0.45))
    trainer, _captured = _build_callback_trainer(
        monkeypatch,
        tmp_path,
        model=model,
        baseline_score=0.5,
        higher_is_better=False,
        data_loader_len=4,
    )

    trainer.train()

    assert (tmp_path / "model" / "state.txt").read_text(encoding="utf-8") == "epoch-1"
    assert model.save_events == [tmp_path / "model", tmp_path / "model"]
    assert trainer.metric_records[-1]["best_epoch"] == 1
    assert trainer.metric_records[-1]["best_dev_metric"] == 0.4


def test_dense_ft_train_method_cli_writes_model_metrics_and_summary(monkeypatch, tmp_path: Path) -> None:
    train_tasks_path = tmp_path / "train.input.json"
    train_labels_path = tmp_path / "train.labels.json"
    train_pairs_path = tmp_path / "train.pairs.json"
    dev_tasks_path = tmp_path / "dev.input.json"
    dev_labels_path = tmp_path / "dev.labels.json"
    output_dir = tmp_path / "dense_ft_run"
    model_dir = output_dir / "checkpoints" / "best_model"
    config_path = tmp_path / "dense_ft_train_stage_config.json"
    train_tasks_path.write_text(json.dumps([_task("train", query="train query")]), encoding="utf-8")
    train_labels_path.write_text(json.dumps([_labels("train")]), encoding="utf-8")
    train_pairs_path.write_text(json.dumps(_pairs("train")), encoding="utf-8")
    dev_tasks_path.write_text(json.dumps([_task("dev", query="dev query")]), encoding="utf-8")
    dev_labels_path.write_text(json.dumps([_labels("dev")]), encoding="utf-8")
    write_dense_ft_train_stage_config(
        config_path,
        train_tasks_path=train_tasks_path,
        train_labels_path=train_labels_path,
        train_pairs_path=train_pairs_path,
        dev_tasks_path=dev_tasks_path,
        dev_labels_path=dev_labels_path,
        output_dir=output_dir,
        model_dir=model_dir,
    )
    monkeypatch.setattr(
        dense_ft_training,
        "_load_sentence_transformer",
        lambda model_name, device: FakeSentenceTransformer(model_name),
    )
    monkeypatch.setattr(
        dense_ft_training,
        "_build_sentence_transformers_fit_runner",
        lambda request: FakeTrainer(request),
    )

    exit_code = train_method_main(
        [
            "--config",
            str(config_path),
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

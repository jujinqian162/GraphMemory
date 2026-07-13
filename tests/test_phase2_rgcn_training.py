import json
import inspect
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch

from graph_memory.models.graph_retriever.batching import build_training_batches
from graph_memory.models.graph_retriever.checkpoint import (
    load_rgcn_checkpoint,
    save_rgcn_checkpoint,
)
from graph_memory.models.dense_finetune.metadata import (
    DenseFinetuneModelMetadata,
    DenseFinetuneSelectionMetadata,
    write_dense_ft_model_metadata,
)
from graph_memory.models.graph_retriever.contracts import TextEmbeddingProvider
from graph_memory.models.graph_retriever.factory import build_model_from_config
from graph_memory.models.graph_retriever.internals.features import (
    NodeFeatureBuilder,
)
from graph_memory.models.graph_retriever.training import (
    RgcnTrainingResult,
    train_graph_retriever,
)
from graph_memory.experiment.config import (
    ArtifactRef,
    DenseEncoderConfig,
    ModelSelectionConfig,
    PairSamplingConfig,
    RgcnModelConfig as ExperimentRgcnModelConfig,
    RgcnTrainConfig,
    RgcnTrainerConfig,
    RgcnDecoderConfig,
    RgcnBeamSearchConfig,
    RgcnBeamLossConfig,
    RgcnOptimizerPhaseConfig,
)
from graph_memory.experiment.invocation import StageInvocation
from graph_memory.experiment.persistence import write_yaml_atomic
from graph_memory.experiment.state import read_stage_summary
from graph_memory.experiment.stage_models import OrdinaryRgcnTrainStageConfig
from graph_memory.retrieval.signals import RetrieverSeedSignalProvider
from graph_memory.contracts.graphs import GraphItemNode, MemoryGraph
from graph_memory.datasets.hotpotqa.projectors import HotpotQAToTextRankingRequest
from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.datasets.hotpotqa.records import (
    HotpotQARankingRecord,
    HotpotQALabelRecord,
)
from graph_memory.contracts.training_pairs import TrainPairRecord
from graph_memory.models.graph_retriever.config.records import (
    NodeFeatureConfig,
    RgcnModelConfig,
    RgcnTrainingConfig,
)
from graph_memory.models.graph_retriever.internals.contracts import GraphBatch
from graph_memory.registry.retrieval import RetrievalMethodId
from graph_memory.stages.train_payloads import RgcnTrainPayload, TrainDependencies
from graph_memory.retrieval.contracts import RankedNode
from graph_memory.retrieval.requests import TextRankingRequest
from graph_memory.stages.train import run_train_stage
from graph_memory.stages.trainers import RgcnGraphRetrieverTrainer
from graph_memory.validation import (
    ContractValidationError,
    validate_graph_batch,
    validate_rgcn_checkpoint_metadata,
    validate_rgcn_model_config,
    validate_training_batch,
)
import scripts.train_method as train_method_script
from scripts.train_method import main as train_method_main


class FakeRetriever:
    method_name = "dense"

    def rank(self, request: TextRankingRequest) -> list[RankedNode]:
        scores = {"m0": 0.9, "m1": 0.2, "m2": 0.7}
        return sorted(
            [
                RankedNode(node_id=candidate.item_id, score=scores[candidate.item_id])
                for candidate in request.candidates
            ],
            key=lambda ranked_node: (-ranked_node.score, ranked_node.node_id),
        )


class FakeTextEmbeddingProvider(TextEmbeddingProvider):
    @property
    def embedding_dim(self) -> int:
        return 4

    def encode_task_nodes(
        self, request: TextRankingRequest, node_ids: list[str]
    ) -> torch.Tensor:
        rows: list[list[float]] = []
        for node_id in node_ids:
            if node_id == "q":
                rows.append([1.0, 0.0, 0.0, 0.0])
            else:
                position = int(node_id[1:])
                rows.append(
                    [
                        0.0,
                        1.0 if position == 0 else 0.0,
                        1.0 if position == 1 else 0.0,
                        1.0 if position == 2 else 0.0,
                    ]
                )
        return torch.tensor(rows, dtype=torch.float32)


def tiny_task_inputs() -> list[HotpotQARankingRecord]:
    return [
        {
            "task_id": "hotpot_rgcn_train",
            "question": "Which evidence mentions Alpha?",
            "candidate_sentences": [
                {
                    "sentence_id": "m0",
                    "text": "Alpha is the answer evidence.",
                    "title": "A",
                    "sentence_index": 0,
                    "position": 0,
                },
                {
                    "sentence_id": "m1",
                    "text": "Beta is unrelated.",
                    "title": "B",
                    "sentence_index": 0,
                    "position": 1,
                },
                {
                    "sentence_id": "m2",
                    "text": "Gamma connects to Alpha.",
                    "title": "C",
                    "sentence_index": 0,
                    "position": 2,
                },
            ],
        }
    ]


def tiny_labels() -> list[HotpotQALabelRecord]:
    return [
        {
            "task_id": "hotpot_rgcn_train",
            "gold_answer": "Alpha",
            "gold_evidence_sentence_ids": ["m0"],
            "gold_dependency_edges": [],
        }
    ]


def tiny_ranking_requests() -> list[TextRankingRequest]:
    projector = HotpotQAToTextRankingRequest()
    return [projector.project(task) for task in tiny_task_inputs()]


def evidence_labels(labels: list[HotpotQALabelRecord]) -> list[EvidenceLabel]:
    return [
        EvidenceLabel(
            task_id=label["task_id"],
            gold_answer=label["gold_answer"],
            gold_evidence_item_ids=tuple(label["gold_evidence_sentence_ids"]),
            gold_dependency_edges=tuple(
                (edge[0], edge[1]) for edge in label["gold_dependency_edges"]
            ),
        )
        for label in labels
    ]


def _graph_nodes(task: HotpotQARankingRecord) -> list[GraphItemNode]:
    return [
        {
            "id": sentence["sentence_id"],
            "node_type": "graph_item",
            "node_kind": "document_sentence",
            "text": sentence["text"],
            "source_ref": sentence["title"],
            "group_key": f"document:{sentence['title']}",
            "sequence_index": sentence["sentence_index"],
            "metadata": {"title": sentence["title"], "position": sentence["position"]},
        }
        for sentence in task["candidate_sentences"]
    ]


def tiny_graphs() -> list[MemoryGraph]:
    task = tiny_task_inputs()[0]
    return [
        {
            "task_id": task["task_id"],
            "nodes": [
                {"id": "q", "node_type": "question", "text": task["question"]},
                *_graph_nodes(task),
            ],
            "edges": [
                {
                    "source": "q",
                    "target": "m0",
                    "edge_type": "query_overlap",
                    "weight": 1.0,
                    "directed": True,
                },
                {
                    "source": "m0",
                    "target": "m2",
                    "edge_type": "bridge",
                    "weight": 0.8,
                    "directed": False,
                },
            ],
        }
    ]


def tiny_pairs() -> list[TrainPairRecord]:
    return [
        {
            "task_id": "hotpot_rgcn_train",
            "node_id": "m0",
            "label": 1,
            "sample_type": "positive",
        },
        {
            "task_id": "hotpot_rgcn_train",
            "node_id": "m1",
            "label": 0,
            "sample_type": "easy_random",
        },
        {
            "task_id": "hotpot_rgcn_train",
            "node_id": "m2",
            "label": 0,
            "sample_type": "hard_graph_neighbor",
        },
    ]


def tiny_model_config() -> RgcnModelConfig:
    return RgcnModelConfig(
        method_name="dense_rgcn_graph_retriever",
        encoder_model="fake-encoder",
        encoder_dim=4,
        query_prefix="query: ",
        passage_prefix="passage: ",
        encoder_batch_size=64,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        feature_config=NodeFeatureConfig(),
        relation_vocab=(
            "query_overlap_forward",
            "sequential_forward",
            "sequential_reverse",
            "entity_overlap_forward",
            "entity_overlap_reverse",
            "bridge_forward",
            "bridge_reverse",
        ),
        graph_encoder_type="rgcn",
        message_transform_type="typed",
        edge_weight_policy="artifact",
        enabled_edge_types=("bridge", "entity_overlap", "query_overlap", "sequential"),
        ablation_name="full_rgcn",
    )


def tiny_training_config() -> RgcnTrainingConfig:
    return RgcnTrainingConfig(
        optimizer_name="AdamW",
        learning_rate=0.01,
        batch_size=1,
        max_grad_norm=1.0,
        random_seed=13,
        pos_weight_enabled=False,
        epochs=2,
    )


def make_rgcn_train_stage_config(
    *,
    train_tasks_path: Path = Path("train.input.json"),
    train_labels_path: Path = Path("train.labels.json"),
    train_graphs_path: Path = Path("train.graphs.json"),
    train_pairs_path: Path = Path("train.pairs.json"),
    dev_tasks_path: Path = Path("dev.input.json"),
    dev_labels_path: Path = Path("dev.labels.json"),
    dev_graphs_path: Path = Path("dev.graphs.json"),
    output_dir: Path = Path("rgcn_run"),
    hidden_dim: int = 8,
    num_layers: int = 1,
    dropout: float = 0.0,
    epochs: int = 1,
    batch_size: int = 1,
    learning_rate: float = 0.01,
    device: str = "cpu",
) -> OrdinaryRgcnTrainStageConfig:
    return OrdinaryRgcnTrainStageConfig(
        stage="train",
        method="dense_rgcn_graph_retriever",
        variant=None,
        dataset="hotpotqa",
        train_tasks=train_tasks_path,
        train_labels=train_labels_path,
        train_graphs=train_graphs_path,
        train_pairs=train_pairs_path,
        dev_tasks=dev_tasks_path,
        dev_labels=dev_labels_path,
        dev_graphs=dev_graphs_path,
        output_dir=output_dir,
        checkpoint_dir=output_dir / "checkpoints",
        metrics=output_dir / "train_metrics.jsonl",
        encoder=DenseEncoderConfig(
            model_name="fake-encoder",
            query_prefix="query: ",
            passage_prefix="passage: ",
            batch_size=64,
        ),
        pairs=PairSamplingConfig(
            random_seed=13,
            easy_random_per_positive=2,
            hard_bm25_per_positive=2,
            hard_dense_per_positive=0,
            hard_graph_neighbor_per_positive=1,
            hard_pool_size=30,
        ),
        train=RgcnTrainConfig(
            model=ExperimentRgcnModelConfig(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                ablation="full_rgcn",
            ),
            trainer=RgcnTrainerConfig(
                optimizer_name="AdamW",
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                max_grad_norm=1.0,
                random_seed=13,
                pos_weight_enabled=False,
                device=device,
            ),
            decoder=RgcnDecoderConfig(
                hidden_dim=hidden_dim,
                step_embedding_dim=16,
                frontier_relation_dim=4,
            ),
            beam=RgcnBeamSearchConfig(
                training_beam_size=2,
                inference_beam_size=2,
                max_steps=5,
                length_penalty_alpha=1.0,
                deduplicate_selected_sets=True,
            ),
            loss=RgcnBeamLossConfig(
                next_action_loss_weight=1.0,
                stop_loss_weight=1.0,
                aux_node_loss_weight=0.2,
            ),
            optimizer_phases=RgcnOptimizerPhaseConfig(
                decoder_warmup_epochs=0,
                decoder_learning_rate=learning_rate,
                rgcn_learning_rate=learning_rate,
            ),
            selection=ModelSelectionConfig(
                best_metric="dev_composite",
                higher_is_better=True,
            ),
        ),
    )


def write_rgcn_train_stage_config(
    path: Path,
    config: OrdinaryRgcnTrainStageConfig,
) -> None:
    invocation = StageInvocation(
        identifier="train:dense_rgcn_graph_retriever",
        stage="train",
        script=Path(train_method_script.__file__).resolve(),
        config_path=path.resolve(),
        summary_path=(config.output_dir / "train.run_summary.yaml").resolve(),
        config=config,
        inputs=tuple(
            ArtifactRef(role=role, path=value.resolve(), kind="file")
            for role, value in (
                ("inputs", config.train_tasks),
                ("labels", config.train_labels),
                ("graphs", config.train_graphs),
                ("train_pairs", config.train_pairs),
                ("dev_inputs", config.dev_tasks),
                ("dev_labels", config.dev_labels),
                ("dev_graphs", config.dev_graphs),
            )
        ),
        outputs=(
            ArtifactRef(
                role="checkpoint",
                path=(config.checkpoint_dir / "best.pt").resolve(),
                kind="file",
            ),
            ArtifactRef(
                role="train_metrics",
                path=config.metrics.resolve(),
                kind="file",
            ),
        ),
        dependencies=(),
        method=RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER,
    )
    write_yaml_atomic(path, invocation)


def _patch_cli_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "graph_memory.stages.trainers._build_rgcn_dependencies",
        lambda _settings, *, device: TrainDependencies(
            text_embedding_provider=FakeTextEmbeddingProvider(),
            seed_signal_provider=RetrieverSeedSignalProvider(FakeRetriever()),
        ),
    )


def test_seed_signal_provider_and_feature_builder_share_rank_semantics():
    provider = RetrieverSeedSignalProvider(FakeRetriever())
    signals = provider.score_task(
        HotpotQAToTextRankingRequest().project(tiny_task_inputs()[0])
    )

    assert [
        (signal.node_id, signal.rank, signal.rank_percentile) for signal in signals
    ] == [
        ("m0", 1, 0.0),
        ("m2", 2, 0.5),
        ("m1", 3, 1.0),
    ]

    builder = NodeFeatureBuilder(NodeFeatureConfig())
    graph_batch = builder.build_node_features(
        node_ids=["q", "m0", "m1", "m2"],
        seed_signals=signals,
    )

    assert graph_batch.node_feature_names == (
        "seed_score",
        "seed_rank_percentile",
        "is_question_node",
    )
    assert graph_batch.scorer_feature_names == ("seed_score", "seed_rank_percentile")
    torch.testing.assert_close(
        graph_batch.node_features,
        torch.tensor(
            [
                [0.0, 1.0, 1.0],
                [0.9, 0.0, 0.0],
                [0.2, 1.0, 0.0],
                [0.7, 0.5, 0.0],
            ],
            dtype=torch.float32,
        ),
    )


def test_model_and_checkpoint_config_validation_reject_missing_dimension():
    valid_config = tiny_model_config()
    validate_rgcn_model_config(valid_config)

    invalid = json.loads(json.dumps(valid_config.to_json_dict()))
    invalid.pop("encoder_dim")
    with pytest.raises(ContractValidationError, match="encoder_dim"):
        validate_rgcn_model_config(invalid)


def test_build_training_batches_uses_dataclasses_not_raw_artifact_dicts():
    batches = build_training_batches(
        ranking_requests=tiny_ranking_requests(),
        graphs=tiny_graphs(),
        pairs=tiny_pairs(),
        model_config=tiny_model_config(),
        text_embedding_provider=FakeTextEmbeddingProvider(),
        seed_signal_provider=RetrieverSeedSignalProvider(FakeRetriever()),
        batch_size=1,
    )

    assert len(batches) == 1
    batch = batches[0]
    assert isinstance(batch.graph_batch, GraphBatch)
    assert batch.sample_node_ids == ["m0", "m1", "m2"]
    assert batch.labels.tolist() == [1.0, 0.0, 0.0]
    assert batch.sample_node_indices.tolist() == [1, 2, 3]
    assert batch.sample_query_indices.tolist() == [0, 0, 0]
    validate_graph_batch(batch.graph_batch)
    validate_training_batch(batch)


def test_train_graph_retriever_writes_metrics_and_best_checkpoint(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"

    def checkpoint_callback(result: RgcnTrainingResult) -> None:
        model = build_model_from_config(result.model_config)
        model.load_state_dict(result.best_model_state_dict)
        save_rgcn_checkpoint(
            checkpoint_dir / "best.pt",
            method_name=result.model_config.method_name,
            model=model,
            optimizer_state_dict=result.optimizer_state_dict,
            scheduler_state_dict=result.scheduler_state_dict,
            epoch=result.best_epoch,
            global_step=result.global_step,
            best_dev_metric=result.best_dev_metric,
            model_config=result.model_config,
            training_config=result.training_config,
        )

    result = train_graph_retriever(
        train_requests=tiny_ranking_requests(),
        train_graphs=tiny_graphs(),
        train_pairs=tiny_pairs(),
        dev_requests=tiny_ranking_requests(),
        dev_labels=evidence_labels(tiny_labels()),
        dev_graphs=tiny_graphs(),
        model_config=tiny_model_config(),
        training_config=tiny_training_config(),
        text_embedding_provider=FakeTextEmbeddingProvider(),
        seed_signal_provider=RetrieverSeedSignalProvider(FakeRetriever()),
        checkpoint_callback=checkpoint_callback,
    )

    assert result.global_step == 2
    assert len(result.metric_records) == 2
    assert result.best_epoch in {1, 2}
    assert result.metric_records[-1]["dev_full_support_at_5"] == 1.0
    negative_counts = result.metric_records[-1]["negative_count_by_type"]
    assert isinstance(negative_counts, dict)
    assert "hard_graph_neighbor" in negative_counts
    for metric_name in (
        "next_action_loss",
        "stop_loss",
        "aux_node_loss",
        "retained_hypotheses",
        "oracle_reachable_rate",
        "premature_stop_rate",
        "average_selected_length",
        "beam_size",
        "max_steps",
    ):
        assert metric_name in result.metric_records[-1]
    assert (checkpoint_dir / "best.pt").exists()

    checkpoint = load_rgcn_checkpoint(
        checkpoint_dir / "best.pt",
        expected_method="dense_rgcn_graph_retriever",
    )
    validate_rgcn_checkpoint_metadata(checkpoint.payload)
    assert checkpoint.model_config == tiny_model_config()


def test_train_graph_retriever_cli_writes_metrics_summary_and_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_cli_dependencies(monkeypatch)
    train_tasks_path = tmp_path / "train.input.json"
    train_labels_path = tmp_path / "train.labels.json"
    train_graphs_path = tmp_path / "train.graphs.json"
    train_pairs_path = tmp_path / "train.pairs.json"
    dev_tasks_path = tmp_path / "dev.input.json"
    dev_labels_path = tmp_path / "dev.labels.json"
    dev_graphs_path = tmp_path / "dev.graphs.json"
    output_dir = tmp_path / "rgcn_run"
    config_path = tmp_path / "rgcn_train_stage_config.yaml"
    train_tasks_path.write_text(json.dumps(tiny_task_inputs()), encoding="utf-8")
    train_labels_path.write_text(json.dumps(tiny_labels()), encoding="utf-8")
    train_graphs_path.write_text(json.dumps(tiny_graphs()), encoding="utf-8")
    train_pairs_path.write_text(json.dumps(tiny_pairs()), encoding="utf-8")
    dev_tasks_path.write_text(json.dumps(tiny_task_inputs()), encoding="utf-8")
    dev_labels_path.write_text(json.dumps(tiny_labels()), encoding="utf-8")
    dev_graphs_path.write_text(json.dumps(tiny_graphs()), encoding="utf-8")
    write_rgcn_train_stage_config(
        config_path,
        make_rgcn_train_stage_config(
            train_tasks_path=train_tasks_path,
            train_labels_path=train_labels_path,
            train_graphs_path=train_graphs_path,
            train_pairs_path=train_pairs_path,
            dev_tasks_path=dev_tasks_path,
            dev_labels_path=dev_labels_path,
            dev_graphs_path=dev_graphs_path,
            output_dir=output_dir,
            hidden_dim=8,
            num_layers=1,
            dropout=0.0,
            epochs=1,
            learning_rate=0.01,
        ),
    )

    exit_code = train_method_main(
        [
            "--config",
            str(config_path),
        ],
    )

    assert exit_code == 0
    assert (output_dir / "train_metrics.jsonl").exists()
    assert (output_dir / "train.run_summary.yaml").exists()
    assert (output_dir / "checkpoints" / "best.pt").exists()
    run_summary = read_stage_summary(output_dir / "train.run_summary.yaml")
    assert run_summary.script.name == "train_method.py"
    assert run_summary.effective_config["method"] == "dense_rgcn_graph_retriever"


def test_train_graph_retriever_cli_reads_model_and_optimization_from_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_cli_dependencies(monkeypatch)
    train_tasks_path = tmp_path / "train.input.json"
    train_labels_path = tmp_path / "train.labels.json"
    train_graphs_path = tmp_path / "train.graphs.json"
    train_pairs_path = tmp_path / "train.pairs.json"
    dev_tasks_path = tmp_path / "dev.input.json"
    dev_labels_path = tmp_path / "dev.labels.json"
    dev_graphs_path = tmp_path / "dev.graphs.json"
    output_dir = tmp_path / "rgcn_run"
    config_path = tmp_path / "rgcn_train_stage_config.yaml"
    train_tasks_path.write_text(json.dumps(tiny_task_inputs()), encoding="utf-8")
    train_labels_path.write_text(json.dumps(tiny_labels()), encoding="utf-8")
    train_graphs_path.write_text(json.dumps(tiny_graphs()), encoding="utf-8")
    train_pairs_path.write_text(json.dumps(tiny_pairs()), encoding="utf-8")
    dev_tasks_path.write_text(json.dumps(tiny_task_inputs()), encoding="utf-8")
    dev_labels_path.write_text(json.dumps(tiny_labels()), encoding="utf-8")
    dev_graphs_path.write_text(json.dumps(tiny_graphs()), encoding="utf-8")
    write_rgcn_train_stage_config(
        config_path,
        make_rgcn_train_stage_config(
            train_tasks_path=train_tasks_path,
            train_labels_path=train_labels_path,
            train_graphs_path=train_graphs_path,
            train_pairs_path=train_pairs_path,
            dev_tasks_path=dev_tasks_path,
            dev_labels_path=dev_labels_path,
            dev_graphs_path=dev_graphs_path,
            output_dir=output_dir,
            hidden_dim=8,
            num_layers=1,
            dropout=0.0,
            epochs=1,
            batch_size=1,
            learning_rate=0.01,
            device="cpu",
        ),
    )

    exit_code = train_method_main(
        [
            "--config",
            str(config_path),
        ],
    )

    assert exit_code == 0
    run_summary = read_stage_summary(output_dir / "train.run_summary.yaml")
    assert run_summary.status == "success"
    train_config = cast(dict[str, object], run_summary.effective_config["train"])
    model_config = cast(dict[str, object], train_config["model"])
    trainer_config = cast(dict[str, object], train_config["trainer"])
    assert model_config["hidden_dim"] == 8
    assert trainer_config["batch_size"] == 1
    assert trainer_config["epochs"] == 1


def test_train_graph_retriever_stage_config_controls_training_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_cli_dependencies(monkeypatch)
    train_tasks_path = tmp_path / "train.input.json"
    train_labels_path = tmp_path / "train.labels.json"
    train_graphs_path = tmp_path / "train.graphs.json"
    train_pairs_path = tmp_path / "train.pairs.json"
    dev_tasks_path = tmp_path / "dev.input.json"
    dev_labels_path = tmp_path / "dev.labels.json"
    dev_graphs_path = tmp_path / "dev.graphs.json"
    output_dir = tmp_path / "rgcn_run"
    config_path = tmp_path / "rgcn_train_stage_config.yaml"
    train_tasks_path.write_text(json.dumps(tiny_task_inputs()), encoding="utf-8")
    train_labels_path.write_text(json.dumps(tiny_labels()), encoding="utf-8")
    train_graphs_path.write_text(json.dumps(tiny_graphs()), encoding="utf-8")
    train_pairs_path.write_text(json.dumps(tiny_pairs()), encoding="utf-8")
    dev_tasks_path.write_text(json.dumps(tiny_task_inputs()), encoding="utf-8")
    dev_labels_path.write_text(json.dumps(tiny_labels()), encoding="utf-8")
    dev_graphs_path.write_text(json.dumps(tiny_graphs()), encoding="utf-8")
    write_rgcn_train_stage_config(
        config_path,
        make_rgcn_train_stage_config(
            train_tasks_path=train_tasks_path,
            train_labels_path=train_labels_path,
            train_graphs_path=train_graphs_path,
            train_pairs_path=train_pairs_path,
            dev_tasks_path=dev_tasks_path,
            dev_labels_path=dev_labels_path,
            dev_graphs_path=dev_graphs_path,
            output_dir=output_dir,
            hidden_dim=8,
            num_layers=1,
            dropout=0.0,
            epochs=1,
            batch_size=1,
            learning_rate=0.01,
            device="cpu",
        ),
    )

    exit_code = train_method_main(
        [
            "--config",
            str(config_path),
        ],
    )

    assert exit_code == 0
    run_summary = read_stage_summary(output_dir / "train.run_summary.yaml")
    train_config = cast(dict[str, object], run_summary.effective_config["train"])
    model_config = cast(dict[str, object], train_config["model"])
    trainer_config = cast(dict[str, object], train_config["trainer"])
    assert model_config["hidden_dim"] == 8
    assert trainer_config["epochs"] == 1
    assert trainer_config["batch_size"] == 1


def test_rgcn_trainer_preserves_dense_ft_seeded_method_identity() -> None:
    config = make_rgcn_train_stage_config().model_copy(
        update={"method": "dense_ft_rgcn_graph_retriever"}
    )

    result = cast(
        RgcnTrainingResult,
        RgcnGraphRetrieverTrainer(config).train(
            RgcnTrainPayload(
                train_requests=tiny_ranking_requests(),
                train_labels=evidence_labels(tiny_labels()),
                train_graphs=tiny_graphs(),
                train_pairs=tiny_pairs(),
                dev_requests=tiny_ranking_requests(),
                dev_labels=evidence_labels(tiny_labels()),
                dev_graphs=tiny_graphs(),
                dependencies=TrainDependencies(
                    text_embedding_provider=FakeTextEmbeddingProvider(),
                    seed_signal_provider=RetrieverSeedSignalProvider(FakeRetriever()),
                ),
            )
        ),
    )

    assert (
        result.model_config.method_name
        == RetrievalMethodId.DENSE_FT_RGCN_GRAPH_RETRIEVER.value
    )


def test_rgcn_trainer_uses_dense_ft_seed_checkpoint_metadata(tmp_path: Path) -> None:
    seed_checkpoint = tmp_path / "dense_ft" / "checkpoints" / "best_model"
    write_dense_ft_model_metadata(
        model_dir=seed_checkpoint,
        metadata=DenseFinetuneModelMetadata(
            base_model="base-e5",
            query_prefix="seed query: ",
            passage_prefix="seed passage: ",
            batch_size=11,
            device="cpu",
            selection=DenseFinetuneSelectionMetadata(
                selected_metric="dev_full_support_at_5",
                higher_is_better=True,
            ),
        ),
    )
    config = make_rgcn_train_stage_config().model_copy(
        update={
            "method": "dense_ft_rgcn_graph_retriever",
            "seed_checkpoint": seed_checkpoint,
        }
    )

    result = cast(
        RgcnTrainingResult,
        RgcnGraphRetrieverTrainer(config).train(
            RgcnTrainPayload(
                train_requests=tiny_ranking_requests(),
                train_labels=evidence_labels(tiny_labels()),
                train_graphs=tiny_graphs(),
                train_pairs=tiny_pairs(),
                dev_requests=tiny_ranking_requests(),
                dev_labels=evidence_labels(tiny_labels()),
                dev_graphs=tiny_graphs(),
                seed_checkpoint=seed_checkpoint,
                dependencies=TrainDependencies(
                    text_embedding_provider=FakeTextEmbeddingProvider(),
                    seed_signal_provider=RetrieverSeedSignalProvider(FakeRetriever()),
                ),
            )
        ),
    )

    assert (
        result.model_config.method_name
        == RetrievalMethodId.DENSE_FT_RGCN_GRAPH_RETRIEVER.value
    )
    assert result.model_config.encoder_model == str(seed_checkpoint)
    assert result.model_config.query_prefix == "seed query: "
    assert result.model_config.passage_prefix == "seed passage: "
    assert result.model_config.encoder_batch_size == 11


def test_dense_ft_seeded_rgcn_checkpoint_validates_selected_method(
    tmp_path: Path,
) -> None:
    model_config = replace(
        tiny_model_config(),
        method_name=RetrievalMethodId.DENSE_FT_RGCN_GRAPH_RETRIEVER.value,
    )
    model = build_model_from_config(model_config)
    checkpoint_path = tmp_path / "best.pt"

    save_rgcn_checkpoint(
        checkpoint_path,
        method_name=RetrievalMethodId.DENSE_FT_RGCN_GRAPH_RETRIEVER.value,
        model=model,
        optimizer_state_dict={},
        scheduler_state_dict={},
        epoch=1,
        global_step=1,
        best_dev_metric=1.0,
        model_config=model_config,
        training_config=tiny_training_config(),
    )

    checkpoint = load_rgcn_checkpoint(
        checkpoint_path,
        expected_method=RetrievalMethodId.DENSE_FT_RGCN_GRAPH_RETRIEVER.value,
    )
    assert (
        checkpoint.model_config.method_name
        == RetrievalMethodId.DENSE_FT_RGCN_GRAPH_RETRIEVER.value
    )
    with pytest.raises(
        ContractValidationError, match="expected_method=dense_rgcn_graph_retriever"
    ):
        load_rgcn_checkpoint(
            checkpoint_path,
            expected_method=RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER.value,
        )


def test_training_registry_builds_trainer_from_settings_type() -> None:
    config = make_rgcn_train_stage_config()

    trainer = RgcnGraphRetrieverTrainer(config)

    assert callable(trainer.train)


def test_train_stage_uses_train_labels_for_pair_validation() -> None:
    config = make_rgcn_train_stage_config()
    mismatched_labels: list[HotpotQALabelRecord] = [
        {
            "task_id": "hotpot_rgcn_train",
            "gold_answer": "Gamma",
            "gold_evidence_sentence_ids": ["m2"],
            "gold_dependency_edges": [],
        }
    ]

    with pytest.raises(
        ContractValidationError, match="positive node_id=m0 is not gold evidence"
    ):
        run_train_stage(
            config,
            payload=RgcnTrainPayload(
                train_requests=tiny_ranking_requests(),
                train_labels=evidence_labels(mismatched_labels),
                train_graphs=tiny_graphs(),
                train_pairs=tiny_pairs(),
                dev_requests=tiny_ranking_requests(),
                dev_labels=evidence_labels(tiny_labels()),
                dev_graphs=tiny_graphs(),
                dependencies=TrainDependencies(
                    text_embedding_provider=FakeTextEmbeddingProvider(),
                    seed_signal_provider=RetrieverSeedSignalProvider(FakeRetriever()),
                ),
            ),
        )


def test_train_stage_runner_and_script_use_registry_boundary() -> None:
    stage_source = Path("graph_memory/stages/train.py").read_text(encoding="utf-8")
    script_source = inspect.getsource(train_method_script)

    assert "RgcnGraphRetrieverTrainer(config).train" in stage_source
    assert "train_graph_retriever" not in stage_source
    assert "assert_never(config)" in stage_source
    assert "dense_rgcn_graph_retriever" not in stage_source
    assert "load_stage_execution(" in script_source
    assert "run_train_stage(config, payload=payload)" in script_source
    assert "load_trainable_training_config" not in script_source
    assert "encoder_config_from_training_config" not in script_source
    assert "model_config_values_from_training_config" not in script_source
    assert "trainable_training_config_from_training_config" not in script_source
    assert not Path("scripts/train_graph_retriever.py").exists()

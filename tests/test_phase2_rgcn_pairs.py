import inspect
import json

import pytest

import graph_memory.training_pairs.builder as pair_builder
from graph_memory.config import CONFIG_LOADER
from graph_memory.contracts.graphs import GraphItemNode, MemoryGraph
from graph_memory.datasets.hotpotqa.projectors import HotpotQAToTextRankingRequest
from graph_memory.datasets.hotpotqa.records import HotpotQARankingRecord, HotpotQALabelRecord
from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.io import write_json
from graph_memory.registry.retrieval import DenseEncoderSettings
from graph_memory.registry.stage_configs import (
    PairBuildIO,
    PairBuildJobSettings,
    PairBuildStageConfig,
    PairSamplingSettings,
)
from graph_memory.retrieval.contracts import RankedNode
from graph_memory.retrieval.requests import TextRankingRequest
from graph_memory.training_pairs import build_train_pairs
from graph_memory.training_pairs.config import NegativeSamplingConfig
from graph_memory.training_pairs.requests import TrainPairBuildTask
from graph_memory.validation import ContractValidationError, validate_train_pairs
import scripts.build_train_pairs as build_train_pairs_script
from scripts.build_train_pairs import main as build_train_pairs_main


def tiny_task_inputs() -> list[HotpotQARankingRecord]:
    return [
        {
            "task_id": "hotpot_pair_test",
            "question": "Which city links the bridge evidence?",
            "candidate_sentences": [
                {
                    "sentence_id": "m0",
                    "text": "Alpha city is connected to the bridge.",
                    "title": "Alpha",
                    "sentence_index": 0,
                    "position": 0,
                },
                {
                    "sentence_id": "m1",
                    "text": "The bridge evidence is in Beta.",
                    "title": "Beta",
                    "sentence_index": 0,
                    "position": 1,
                },
                {
                    "sentence_id": "m2",
                    "text": "Gamma is a nearby distractor.",
                    "title": "Gamma",
                    "sentence_index": 0,
                    "position": 2,
                },
                {
                    "sentence_id": "m3",
                    "text": "The answer depends on Delta.",
                    "title": "Delta",
                    "sentence_index": 0,
                    "position": 3,
                },
            ],
        }
    ]


def tiny_labels() -> list[HotpotQALabelRecord]:
    return [
        {
            "task_id": "hotpot_pair_test",
            "gold_answer": "Beta and Delta",
            "gold_evidence_sentence_ids": ["m1", "m3"],
            "gold_dependency_edges": [],
        }
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
                {"source": "m1", "target": "m2", "edge_type": "bridge", "weight": 1.0, "directed": False},
                {"source": "m3", "target": "m0", "edge_type": "entity_overlap", "weight": 1.0, "directed": False},
            ],
        }
    ]


def _ranking_requests():
    projector = HotpotQAToTextRankingRequest()
    return [projector.project(task) for task in tiny_task_inputs()]


def _evidence_labels():
    return [
        EvidenceLabel(
            task_id=label["task_id"],
            gold_answer=label["gold_answer"],
            gold_evidence_item_ids=tuple(label["gold_evidence_sentence_ids"]),
            gold_dependency_edges=tuple((edge[0], edge[1]) for edge in label["gold_dependency_edges"]),
        )
        for label in tiny_labels()
    ]


def _label_by_task_id():
    return {label.task_id: label for label in _evidence_labels()}


def _pair_tasks() -> list[TrainPairBuildTask]:
    return [
        TrainPairBuildTask(text_request=request, label=label, graph=graph)
        for request, label, graph in zip(_ranking_requests(), _evidence_labels(), tiny_graphs(), strict=True)
    ]


def by_task_id(records):
    return {record["task_id"]: record for record in records}


def write_pair_stage_config(
    path,
    *,
    tasks_path,
    labels_path,
    graphs_path,
    output_path,
    sampling: PairSamplingSettings,
    hard_dense_encoder: DenseEncoderSettings | None = None,
) -> None:
    config = PairBuildStageConfig(
        io=PairBuildIO(
            tasks=tasks_path,
            labels=labels_path,
            graphs=graphs_path,
            output=output_path,
            summary=output_path.with_name("train_pairs.summary.json"),
            run_summary=output_path.with_name("train_pairs.run_summary.json"),
        ),
        job=PairBuildJobSettings(
            sampling=sampling,
            hard_dense_encoder=hard_dense_encoder,
        ),
    )
    write_json(path, CONFIG_LOADER.to_json(config))


def pair_sampling(
    *,
    easy_random_per_positive: int,
    hard_bm25_per_positive: int,
    hard_dense_per_positive: int,
    hard_graph_neighbor_per_positive: int,
    random_seed: int = 7,
    hard_pool_size: int = 10,
) -> PairSamplingSettings:
    return PairSamplingSettings(
        random_seed=random_seed,
        easy_random_per_positive=easy_random_per_positive,
        hard_bm25_per_positive=hard_bm25_per_positive,
        hard_dense_per_positive=hard_dense_per_positive,
        hard_graph_neighbor_per_positive=hard_graph_neighbor_per_positive,
        hard_pool_size=hard_pool_size,
    )


def test_train_pair_validation_rejects_question_node_sample():
    pairs = [
        {"task_id": "hotpot_pair_test", "node_id": "m1", "label": 1, "sample_type": "positive"},
        {"task_id": "hotpot_pair_test", "node_id": "q", "label": 0, "sample_type": "easy_random"},
    ]

    with pytest.raises(ContractValidationError, match="q"):
        validate_train_pairs(pairs, _ranking_requests(), _label_by_task_id(), by_task_id(tiny_graphs()))


def test_build_train_pairs_creates_valid_positive_random_and_graph_neighbor_samples():
    config = NegativeSamplingConfig(
        random_seed=7,
        easy_random_per_positive=1,
        hard_bm25_per_positive=0,
        hard_dense_per_positive=0,
        hard_graph_neighbor_per_positive=1,
        hard_pool_size=10,
    )

    result = build_train_pairs(_pair_tasks(), config)

    assert result.summary["positive_count"] == 2
    assert result.summary["negative_count_by_type"] == {"easy_random": 2, "hard_graph_neighbor": 2}
    assert {pair["node_id"] for pair in result.pairs if pair["label"] == 1} == {"m1", "m3"}
    assert all(pair["node_id"] != "q" for pair in result.pairs)
    validate_train_pairs(
        result.pairs,
        _ranking_requests(),
        _label_by_task_id(),
        by_task_id(tiny_graphs()),
    )


def test_build_train_pairs_cli_writes_pairs_summary_and_run_summary(tmp_path):
    tasks_path = tmp_path / "train_memory_tasks.input.json"
    labels_path = tmp_path / "train_memory_tasks.labels.json"
    graphs_path = tmp_path / "train_graphs.json"
    output_path = tmp_path / "train_pairs.json"
    config_path = tmp_path / "pair_stage_config.json"
    tasks_path.write_text(json.dumps(tiny_task_inputs()), encoding="utf-8")
    labels_path.write_text(json.dumps(tiny_labels()), encoding="utf-8")
    graphs_path.write_text(json.dumps(tiny_graphs()), encoding="utf-8")
    write_pair_stage_config(
        config_path,
        tasks_path=tasks_path,
        labels_path=labels_path,
        graphs_path=graphs_path,
        output_path=output_path,
        sampling=pair_sampling(
            easy_random_per_positive=1,
            hard_bm25_per_positive=0,
            hard_dense_per_positive=0,
            hard_graph_neighbor_per_positive=1,
        ),
    )

    exit_code = build_train_pairs_main(
        [
            "--config",
            str(config_path),
        ]
    )

    assert exit_code == 0
    pairs = json.loads(output_path.read_text(encoding="utf-8"))
    summary = json.loads(output_path.with_name("train_pairs.summary.json").read_text(encoding="utf-8"))
    run_summary = json.loads(output_path.with_name("train_pairs.run_summary.json").read_text(encoding="utf-8"))
    assert summary["positive_count"] == 2
    assert summary["negative_count_by_type"]["hard_graph_neighbor"] == 2
    assert run_summary["status"] == "success"
    assert all(pair["node_id"] != "q" for pair in pairs)


def test_build_train_pairs_cli_reads_pair_sampling_from_config(tmp_path):
    tasks_path = tmp_path / "train_memory_tasks.input.json"
    labels_path = tmp_path / "train_memory_tasks.labels.json"
    graphs_path = tmp_path / "train_graphs.json"
    output_path = tmp_path / "train_pairs.json"
    config_path = tmp_path / "pair_stage_config.json"
    tasks_path.write_text(json.dumps(tiny_task_inputs()), encoding="utf-8")
    labels_path.write_text(json.dumps(tiny_labels()), encoding="utf-8")
    graphs_path.write_text(json.dumps(tiny_graphs()), encoding="utf-8")
    write_pair_stage_config(
        config_path,
        tasks_path=tasks_path,
        labels_path=labels_path,
        graphs_path=graphs_path,
        output_path=output_path,
        sampling=pair_sampling(
            easy_random_per_positive=1,
            hard_bm25_per_positive=0,
            hard_dense_per_positive=0,
            hard_graph_neighbor_per_positive=1,
        ),
    )

    exit_code = build_train_pairs_main(
        [
            "--config",
            str(config_path),
        ]
    )

    assert exit_code == 0
    summary = json.loads(output_path.with_name("train_pairs.summary.json").read_text(encoding="utf-8"))
    run_summary = json.loads(output_path.with_name("train_pairs.run_summary.json").read_text(encoding="utf-8"))
    assert summary["negative_count_by_type"] == {"easy_random": 2, "hard_graph_neighbor": 2}
    assert run_summary["effective_config"]["hard_dense_per_positive"] == 0
    assert run_summary["effective_config"]["hard_graph_neighbor_per_positive"] == 1


def test_build_train_pairs_cli_uses_config_encoder_for_hard_dense_negatives(tmp_path, monkeypatch):
    tasks_path = tmp_path / "train_memory_tasks.input.json"
    labels_path = tmp_path / "train_memory_tasks.labels.json"
    graphs_path = tmp_path / "train_graphs.json"
    output_path = tmp_path / "train_pairs.json"
    config_path = tmp_path / "pair_stage_config.json"
    observed_model_names = []
    tasks_path.write_text(json.dumps(tiny_task_inputs()), encoding="utf-8")
    labels_path.write_text(json.dumps(tiny_labels()), encoding="utf-8")
    graphs_path.write_text(json.dumps(tiny_graphs()), encoding="utf-8")
    write_pair_stage_config(
        config_path,
        tasks_path=tasks_path,
        labels_path=labels_path,
        graphs_path=graphs_path,
        output_path=output_path,
        sampling=pair_sampling(
            easy_random_per_positive=0,
            hard_bm25_per_positive=0,
            hard_dense_per_positive=1,
            hard_graph_neighbor_per_positive=0,
        ),
        hard_dense_encoder=DenseEncoderSettings(
            model_name="models/local-e5",
            query_prefix="query: ",
            passage_prefix="passage: ",
            batch_size=64,
        ),
    )

    class FakeDenseTaskRetriever:
        def __init__(
            self,
            model_name="intfloat/e5-base-v2",
            batch_size=64,
            query_prefix="query: ",
            passage_prefix="passage: ",
            config=None,
            encoder=None,
        ):
            observed_model_names.append(model_name if config is None else config.model_name)

        def rank(self, request: TextRankingRequest):
            return [
                RankedNode(node_id=candidate.item_id, score=float(index))
                for index, candidate in enumerate(request.candidates)
            ]

    monkeypatch.setattr(pair_builder, "DenseTaskRetriever", FakeDenseTaskRetriever)

    exit_code = build_train_pairs_main(
        [
            "--config",
            str(config_path),
        ]
    )

    assert exit_code == 0
    assert observed_model_names == ["models/local-e5"]


def test_build_train_pairs_stage_config_controls_sampling_without_cli_overrides(tmp_path, monkeypatch):
    tasks_path = tmp_path / "train_memory_tasks.input.json"
    labels_path = tmp_path / "train_memory_tasks.labels.json"
    graphs_path = tmp_path / "train_graphs.json"
    output_path = tmp_path / "train_pairs.json"
    config_path = tmp_path / "pair_stage_config.json"
    observed_model_names = []
    tasks_path.write_text(json.dumps(tiny_task_inputs()), encoding="utf-8")
    labels_path.write_text(json.dumps(tiny_labels()), encoding="utf-8")
    graphs_path.write_text(json.dumps(tiny_graphs()), encoding="utf-8")
    write_pair_stage_config(
        config_path,
        tasks_path=tasks_path,
        labels_path=labels_path,
        graphs_path=graphs_path,
        output_path=output_path,
        sampling=pair_sampling(
            easy_random_per_positive=1,
            hard_bm25_per_positive=0,
            hard_dense_per_positive=0,
            hard_graph_neighbor_per_positive=0,
        ),
    )

    class FakeDenseTaskRetriever:
        def __init__(
            self,
            model_name="intfloat/e5-base-v2",
            batch_size=64,
            query_prefix="query: ",
            passage_prefix="passage: ",
            config=None,
            encoder=None,
        ):
            observed_model_names.append(model_name if config is None else config.model_name)

        def rank(self, request: TextRankingRequest):
            return [
                RankedNode(node_id=candidate.item_id, score=float(index))
                for index, candidate in enumerate(request.candidates)
            ]

    monkeypatch.setattr(pair_builder, "DenseTaskRetriever", FakeDenseTaskRetriever)

    exit_code = build_train_pairs_main(
        [
            "--config",
            str(config_path),
        ]
    )

    assert exit_code == 0
    summary = json.loads(output_path.with_name("train_pairs.summary.json").read_text(encoding="utf-8"))
    run_summary = json.loads(output_path.with_name("train_pairs.run_summary.json").read_text(encoding="utf-8"))
    assert summary["negative_count_by_type"] == {"easy_random": 2}
    assert run_summary["effective_config"]["easy_random_per_positive"] == 1
    assert run_summary["effective_config"]["hard_dense_per_positive"] == 0
    assert observed_model_names == []


def test_build_train_pairs_config_hard_dense_requires_config_encoder(tmp_path, monkeypatch):
    tasks_path = tmp_path / "train_memory_tasks.input.json"
    labels_path = tmp_path / "train_memory_tasks.labels.json"
    graphs_path = tmp_path / "train_graphs.json"
    output_path = tmp_path / "train_pairs.json"
    config_path = tmp_path / "pair_stage_config.json"
    tasks_path.write_text(json.dumps(tiny_task_inputs()), encoding="utf-8")
    labels_path.write_text(json.dumps(tiny_labels()), encoding="utf-8")
    graphs_path.write_text(json.dumps(tiny_graphs()), encoding="utf-8")
    write_pair_stage_config(
        config_path,
        tasks_path=tasks_path,
        labels_path=labels_path,
        graphs_path=graphs_path,
        output_path=output_path,
        sampling=pair_sampling(
            easy_random_per_positive=0,
            hard_bm25_per_positive=0,
            hard_dense_per_positive=1,
            hard_graph_neighbor_per_positive=0,
        ),
    )

    class UnexpectedDenseTaskRetriever:
        def __init__(self, *args, **kwargs):
            raise AssertionError("default dense retriever should not be loaded in config mode")

    monkeypatch.setattr(pair_builder, "DenseTaskRetriever", UnexpectedDenseTaskRetriever)

    with pytest.raises(ValueError, match="encoder"):
        build_train_pairs_main(
            [
                "--config",
                str(config_path),
            ]
        )


def test_build_train_pairs_script_uses_pair_stage_registry_without_training_dict_slicing_helpers():
    source = inspect.getsource(build_train_pairs_script)

    assert "Registry.configs.PAIRS" in source
    assert "load_trainable_training_config" not in source
    assert "negative_sampling_config_from_training_config" not in source
    assert "encoder_config_from_training_config" not in source

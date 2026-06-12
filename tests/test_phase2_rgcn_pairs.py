import inspect
import json

import pytest

import graph_memory.training_pairs.builder as pair_builder
from graph_memory.config import CONFIG_LOADER
from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.tasks import MemoryTaskInput, MemoryTaskLabels
from graph_memory.io import write_json
from graph_memory.registry.retrieval import DenseEncoderSettings
from graph_memory.registry.stage_configs import (
    PairBuildIO,
    PairBuildJobSettings,
    PairBuildStageConfig,
    PairSamplingSettings,
)
from graph_memory.retrieval.contracts import RankedNode
from graph_memory.training_pairs import build_train_pairs
from graph_memory.training_pairs.config import NegativeSamplingConfig
from graph_memory.validation import ContractValidationError, validate_train_pairs
import scripts.build_train_pairs as build_train_pairs_script
from scripts.build_train_pairs import main as build_train_pairs_main


def tiny_task_inputs() -> list[MemoryTaskInput]:
    return [
        {
            "task_id": "hotpot_pair_test",
            "query": "Which city links the bridge evidence?",
            "memory_items": [
                {
                    "id": "m0",
                    "node_type": "document_sentence",
                    "text": "Alpha city is connected to the bridge.",
                    "source": "Alpha",
                    "sentence_id": 0,
                    "position": 0,
                },
                {
                    "id": "m1",
                    "node_type": "document_sentence",
                    "text": "The bridge evidence is in Beta.",
                    "source": "Beta",
                    "sentence_id": 0,
                    "position": 1,
                },
                {
                    "id": "m2",
                    "node_type": "document_sentence",
                    "text": "Gamma is a nearby distractor.",
                    "source": "Gamma",
                    "sentence_id": 0,
                    "position": 2,
                },
                {
                    "id": "m3",
                    "node_type": "document_sentence",
                    "text": "The answer depends on Delta.",
                    "source": "Delta",
                    "sentence_id": 0,
                    "position": 3,
                },
            ],
        }
    ]


def tiny_labels() -> list[MemoryTaskLabels]:
    return [
        {
            "task_id": "hotpot_pair_test",
            "gold_answer": "Beta and Delta",
            "gold_evidence_nodes": ["m1", "m3"],
            "gold_dependency_edges": [],
        }
    ]


def tiny_graphs() -> list[MemoryGraph]:
    task = tiny_task_inputs()[0]
    return [
        {
            "task_id": task["task_id"],
            "nodes": [
                {"id": "q", "node_type": "question", "text": task["query"]},
                *task["memory_items"],
            ],
            "edges": [
                {"source": "m1", "target": "m2", "edge_type": "bridge", "weight": 1.0, "directed": False},
                {"source": "m3", "target": "m0", "edge_type": "entity_overlap", "weight": 1.0, "directed": False},
            ],
        }
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
        validate_train_pairs(pairs, by_task_id(tiny_task_inputs()), by_task_id(tiny_labels()), by_task_id(tiny_graphs()))


def test_build_train_pairs_creates_valid_positive_random_and_graph_neighbor_samples():
    config = NegativeSamplingConfig(
        random_seed=7,
        easy_random_per_positive=1,
        hard_bm25_per_positive=0,
        hard_dense_per_positive=0,
        hard_graph_neighbor_per_positive=1,
        hard_pool_size=10,
    )

    result = build_train_pairs(tiny_task_inputs(), tiny_labels(), tiny_graphs(), config)

    assert result.summary["positive_count"] == 2
    assert result.summary["negative_count_by_type"] == {"easy_random": 2, "hard_graph_neighbor": 2}
    assert {pair["node_id"] for pair in result.pairs if pair["label"] == 1} == {"m1", "m3"}
    assert all(pair["node_id"] != "q" for pair in result.pairs)
    validate_train_pairs(
        result.pairs,
        by_task_id(tiny_task_inputs()),
        by_task_id(tiny_labels()),
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

        def rank(self, task_input):
            return [
                RankedNode(node_id=memory_item["id"], score=float(index))
                for index, memory_item in enumerate(task_input["memory_items"])
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

        def rank(self, task_input):
            return [
                RankedNode(node_id=memory_item["id"], score=float(index))
                for index, memory_item in enumerate(task_input["memory_items"])
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

from __future__ import annotations

import json
from pathlib import Path

from graph_memory.io import read_json, write_json
from scripts import build_proposal_graphs
from scripts.workflow import (
    build_stage_plan,
    initialize_experiment,
    inspect_experiment_status,
    load_experiment_config,
    prune_manifest_completed_prefix,
)
from scripts.workflow.types import ArtifactState, StageId


LEARNED = "learned_graph_rgcn_retriever"
RGCN = "dense_rgcn_graph_retriever"


def _config_with_learned_method() -> dict[str, object]:
    config = load_experiment_config()
    config["method_configs"] = {
        **config.get("method_configs", {}),
        LEARNED: "configs/methods/learned_graph_rgcn_retriever.json",
    }
    return config


def _twowiki_task() -> dict[str, object]:
    return {
        "task_id": "2wiki_abc123",
        "question": "Who is Ada's mother?",
        "question_type": "compositional",
        "candidate_sentences": [
            {
                "sentence_id": "m0",
                "title": "Film A",
                "sentence_index": 0,
                "position": 0,
                "text": "Film A was directed by Ada.",
            },
            {
                "sentence_id": "m1",
                "title": "Ada Lovelace",
                "sentence_index": 0,
                "position": 1,
                "text": "Ada was the daughter of Beth.",
            },
        ],
        "metadata": {"dataset": "2wiki", "raw_id": "abc123"},
    }


def test_build_proposal_graphs_cli_uses_dataset_selector_and_high_recall_defaults(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.json"
    graphs_path = tmp_path / "proposal.graphs.json"
    write_json(tasks_path, [_twowiki_task()])

    assert build_proposal_graphs.main(
        [
            "--dataset",
            "twowiki",
            "--method",
            LEARNED,
            "--input",
            str(tasks_path),
            "--output",
            str(graphs_path),
        ]
    ) == 0

    graphs = read_json(graphs_path)
    summary = read_json(graphs_path.with_name("proposal.graphs.run_summary.json"))
    graph_payload = json.dumps(graphs, sort_keys=True)

    assert graphs[0]["task_id"] == "2wiki_abc123"
    assert summary["effective_config"]["method"] == LEARNED
    assert summary["effective_config"]["max_query_overlap"] == 80
    assert summary["effective_config"]["max_entity_neighbors"] == 30
    assert summary["effective_config"]["max_bridge_edges"] == 200
    assert "gold_dependency_edges" not in graph_payload
    assert "supporting_facts" not in graph_payload
    assert "evidences" not in graph_payload
    assert "answer" not in graph_payload


def test_learned_rgcn_uses_method_local_proposal_graphs_without_shared_graph_stage(tmp_path: Path) -> None:
    manifest = initialize_experiment(
        "learned-only",
        config=_config_with_learned_method(),
        run_root=tmp_path,
        profile="smoke",
        methods=[LEARNED],
        force=True,
    )

    assert set(manifest["artifacts"]["proposal_graphs"]) == {LEARNED}
    assert manifest["artifacts"]["proposal_graphs"][LEARNED]["train"].endswith(
        "graphs/train.learned_graph_rgcn_retriever.graphs.json"
    )
    assert "graphs" in manifest["artifacts"]

    pair_config = read_json(Path(manifest["stage_configs"]["pairs"][LEARNED]))
    train_config = read_json(Path(manifest["stage_configs"]["train"][LEARNED]))
    retrieve_config = read_json(Path(manifest["stage_configs"]["retrieve"][LEARNED]))
    evaluate_config = read_json(Path(manifest["stage_configs"]["evaluate"][LEARNED]))

    assert Path(pair_config["io"]["graphs"]) == Path(manifest["artifacts"]["proposal_graphs"][LEARNED]["train"])
    assert Path(train_config["io"]["train_graphs"]) == Path(manifest["artifacts"]["proposal_graphs"][LEARNED]["train"])
    assert Path(train_config["io"]["dev_graphs"]) == Path(manifest["artifacts"]["proposal_graphs"][LEARNED]["dev"])
    assert Path(retrieve_config["io"]["graphs"]) == Path(manifest["artifacts"]["proposal_graphs"][LEARNED]["test"])
    assert Path(evaluate_config["io"]["graphs"]) == Path(manifest["artifacts"]["proposal_graphs"][LEARNED]["test"])

    commands = build_stage_plan(
        manifest,
        from_stage="prepare",
        to_stage="pairs",
        methods=[LEARNED],
    )

    assert StageId.GRAPHS not in {command.stage for command in commands}
    assert [command.stage for command in commands].count(StageId.PROPOSAL_GRAPHS) == 3
    proposal_commands = [command for command in commands if command.stage is StageId.PROPOSAL_GRAPHS]
    assert {command.method for command in proposal_commands} == {LEARNED}
    assert {Path(command.argv[command.argv.index("--output") + 1]).name for command in proposal_commands} == {
        "train.learned_graph_rgcn_retriever.graphs.json",
        "dev.learned_graph_rgcn_retriever.graphs.json",
        "test.learned_graph_rgcn_retriever.graphs.json",
    }


def test_active_hotpotqa_config_wires_learned_graph_method_to_proposal_graphs(tmp_path: Path) -> None:
    config = load_experiment_config("hotpotqa_evidence_retrieval")

    assert LEARNED in config["methods"]
    assert config["method_configs"][LEARNED] == "configs/methods/learned_graph_rgcn_retriever.json"

    manifest = initialize_experiment(
        "hotpotqa-learned",
        config=config,
        run_root=tmp_path,
        profile="smoke",
        methods=[LEARNED],
        force=True,
    )
    commands = build_stage_plan(
        manifest,
        from_stage="prepare",
        to_stage="evaluate",
        methods=[LEARNED],
    )
    train_config = read_json(Path(manifest["stage_configs"]["train"][LEARNED]))
    retrieve_config = read_json(Path(manifest["stage_configs"]["retrieve"][LEARNED]))

    assert manifest["effective_config"]["resolved_method_configs"][LEARNED]["method"] == LEARNED
    assert train_config["job"]["loss"] == {
        "rank_loss_weight": 1.0,
        "edge_loss_weight": 0.2,
        "sparse_loss_weight": 0.05,
    }
    assert Path(train_config["io"]["train_graphs"]) == Path(manifest["artifacts"]["proposal_graphs"][LEARNED]["train"])
    assert Path(retrieve_config["io"]["graphs"]) == Path(manifest["artifacts"]["proposal_graphs"][LEARNED]["test"])
    assert StageId.GRAPHS not in {command.stage for command in commands}
    assert {command.stage for command in commands} >= {
        StageId.PREPARE,
        StageId.PROPOSAL_GRAPHS,
        StageId.PAIRS,
        StageId.TRAIN,
        StageId.RETRIEVE,
        StageId.EVALUATE,
    }


def test_learned_and_existing_rgcn_graph_chains_are_parallel(tmp_path: Path) -> None:
    manifest = initialize_experiment(
        "learned-with-existing-rgcn",
        config=_config_with_learned_method(),
        run_root=tmp_path,
        profile="smoke",
        methods=[RGCN, LEARNED],
        force=True,
    )

    rgcn_train = read_json(Path(manifest["stage_configs"]["train"][RGCN]))
    learned_train = read_json(Path(manifest["stage_configs"]["train"][LEARNED]))
    rgcn_retrieve = read_json(Path(manifest["stage_configs"]["retrieve"][RGCN]))
    learned_retrieve = read_json(Path(manifest["stage_configs"]["retrieve"][LEARNED]))

    assert Path(rgcn_train["io"]["train_graphs"]) == Path(manifest["artifacts"]["graphs"]["train"])
    assert Path(rgcn_retrieve["io"]["graphs"]) == Path(manifest["artifacts"]["graphs"]["test"])
    assert Path(learned_train["io"]["train_graphs"]) == Path(manifest["artifacts"]["proposal_graphs"][LEARNED]["train"])
    assert Path(learned_retrieve["io"]["graphs"]) == Path(manifest["artifacts"]["proposal_graphs"][LEARNED]["test"])

    commands = build_stage_plan(
        manifest,
        from_stage="graphs",
        to_stage="pairs",
        methods=[RGCN, LEARNED],
    )

    graph_commands = [command for command in commands if command.stage is StageId.GRAPHS]
    proposal_commands = [command for command in commands if command.stage is StageId.PROPOSAL_GRAPHS]
    assert len(graph_commands) == 3
    assert len(proposal_commands) == 3
    assert {command.method for command in proposal_commands} == {LEARNED}
    assert all(
        Path(command.argv[command.argv.index("--output") + 1]).name.endswith(
            ".learned_graph_rgcn_retriever.graphs.json"
        )
        for command in proposal_commands
    )


def test_learned_status_and_cache_use_proposal_graph_artifacts(tmp_path: Path) -> None:
    manifest = initialize_experiment(
        "learned-status",
        config=_config_with_learned_method(),
        run_root=tmp_path,
        profile="smoke",
        methods=[LEARNED],
        force=True,
    )

    for split, path in manifest["artifacts"]["proposal_graphs"][LEARNED].items():
        graph_path = Path(path)
        write_json(graph_path, [])
        write_json(
            graph_path.with_name(f"{graph_path.stem}.run_summary.json"),
            {
                "script": "build_proposal_graphs.py",
                "status": "success",
                "inputs": {"tasks": manifest["artifacts"]["inputs"][split]["input"]},
                "outputs": {
                    "proposal_graphs": path,
                    "graph_stats": graph_path.with_name(f"{graph_path.stem}.stats.json").as_posix(),
                    "run_summary": graph_path.with_name(f"{graph_path.stem}.run_summary.json").as_posix(),
                },
                "effective_config": {
                    "dataset": "hotpotqa",
                    "method": LEARNED,
                    **manifest["effective_config"]["resolved_method_configs"][LEARNED]["proposal_graph"],
                },
            },
        )

    learned = manifest["artifacts"]["learned"][LEARNED]
    write_json(learned["train_pairs"], [])
    write_json(learned["train_pair_summary"], {})
    write_json(
        learned["train_pair_run_summary"],
        {
            "script": "build_train_pairs.py",
            "status": "success",
            "inputs": {
                "tasks": manifest["artifacts"]["inputs"]["train"]["input"],
                "labels": manifest["artifacts"]["inputs"]["train"]["labels"],
                "graphs": manifest["artifacts"]["proposal_graphs"][LEARNED]["train"],
            },
            "outputs": {
                "pairs": learned["train_pairs"],
                "summary": learned["train_pair_summary"],
                "run_summary": learned["train_pair_run_summary"],
            },
            "effective_config": {
                "dataset": "hotpotqa",
                **manifest["effective_config"]["resolved_method_configs"][LEARNED]["pairs"],
            },
        },
    )
    checkpoint_path = Path(learned["best_checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"checkpoint")
    Path(learned["train_metrics"]).parent.mkdir(parents=True, exist_ok=True)
    Path(learned["train_metrics"]).write_text("{}\n", encoding="utf-8")
    write_json(
        learned["train_run_summary"],
        {
            "script": "train_method.py",
            "status": "success",
            "inputs": {
                "train_tasks": manifest["artifacts"]["inputs"]["train"]["input"],
                "train_labels": manifest["artifacts"]["inputs"]["train"]["labels"],
                "train_pairs": learned["train_pairs"],
                "train_graphs": manifest["artifacts"]["proposal_graphs"][LEARNED]["train"],
                "dev_tasks": manifest["artifacts"]["inputs"]["dev"]["input"],
                "dev_labels": manifest["artifacts"]["inputs"]["dev"]["labels"],
                "dev_graphs": manifest["artifacts"]["proposal_graphs"][LEARNED]["dev"],
            },
            "outputs": {
                "best_checkpoint": learned["best_checkpoint"],
                "metrics": learned["train_metrics"],
                "run_summary": learned["train_run_summary"],
            },
            "effective_config": {"dataset": "hotpotqa", "method": LEARNED},
        },
    )
    prediction_path = Path(manifest["artifacts"]["predictions"][LEARNED])
    write_json(prediction_path, [])
    write_json(
        prediction_path.with_name(f"{prediction_path.stem}.run_summary.json"),
        {
            "script": "run_retrieval.py",
            "status": "success",
            "inputs": {"tasks": manifest["artifacts"]["inputs"]["test"]["input"]},
            "outputs": {"predictions": prediction_path.as_posix()},
            "effective_config": {"dataset": "hotpotqa", "method": LEARNED, "top_k": 10},
        },
    )
    metric_path = Path(manifest["artifacts"]["metrics"][LEARNED])
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.write_text("method,Recall@10\nlearned_graph_rgcn_retriever,1\n", encoding="utf-8")
    failure_cases = Path(manifest["artifacts"]["failure_cases"][LEARNED])
    failure_cases.parent.mkdir(parents=True, exist_ok=True)
    failure_cases.write_text("", encoding="utf-8")
    write_json(
        metric_path.with_name(f"{metric_path.stem}.run_summary.json"),
        {
            "script": "evaluate_retrieval.py",
            "status": "success",
            "inputs": {
                "predictions": manifest["artifacts"]["predictions"][LEARNED],
                "labels": manifest["artifacts"]["inputs"]["test"]["labels"],
                "graphs": manifest["artifacts"]["proposal_graphs"][LEARNED]["test"],
            },
            "outputs": {
                "metrics": manifest["artifacts"]["metrics"][LEARNED],
                "failure_cases": manifest["artifacts"]["failure_cases"][LEARNED],
            },
            "effective_config": {"dataset": "hotpotqa", "failure_case_limit": 50},
        },
    )

    rows = inspect_experiment_status(manifest)
    state_by_key = {
        (row["stage"], row.get("method"), row.get("split")): row["state"]
        for row in rows
    }

    assert all(row["stage"] != StageId.GRAPHS.value for row in rows)
    assert state_by_key[(StageId.PROPOSAL_GRAPHS.value, LEARNED, "train")] == ArtifactState.COMPLETE.value
    assert state_by_key[(StageId.PROPOSAL_GRAPHS.value, LEARNED, "dev")] == ArtifactState.COMPLETE.value
    assert state_by_key[(StageId.PROPOSAL_GRAPHS.value, LEARNED, "test")] == ArtifactState.COMPLETE.value
    assert state_by_key[(StageId.PAIRS.value, LEARNED, None)] == ArtifactState.COMPLETE.value
    assert state_by_key[(StageId.TRAIN.value, LEARNED, None)] == ArtifactState.COMPLETE.value
    assert state_by_key[(StageId.EVALUATE.value, LEARNED, None)] == ArtifactState.COMPLETE.value

    commands = build_stage_plan(
        manifest,
        from_stage="proposal_graphs",
        to_stage="evaluate",
        methods=[LEARNED],
    )
    decision = prune_manifest_completed_prefix(manifest, commands)

    assert decision.commands == ()
    assert [command.stage for command in decision.skipped] == [
        StageId.PROPOSAL_GRAPHS,
        StageId.PROPOSAL_GRAPHS,
        StageId.PROPOSAL_GRAPHS,
        StageId.PAIRS,
        StageId.TRAIN,
        StageId.RETRIEVE,
        StageId.EVALUATE,
    ]

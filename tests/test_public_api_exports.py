from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    ("module_name", "expected"),
    [
        ("graph_memory.config", {"CONFIG_LOADER", "ConfigLoader"}),
        ("graph_memory.registry", {"Registry"}),
        ("graph_memory.training_pairs", {"build_train_pairs"}),
        (
            "graph_memory.retrieval.tuning",
            {"graph_rerank_grid", "graph_rerank_grid_from_record", "tune_graph_rerank"},
        ),
        (
            "graph_memory.io",
            {"merge_config", "read_csv", "read_json", "write_csv", "write_json", "write_jsonl"},
        ),
        (
            "graph_memory.training_config",
            {"device_from_training_config", "load_trainable_training_config"},
        ),
        (
            "graph_memory.registry.stage_configs",
            {
                "EvaluateStageConfig",
                "PairBuildStageConfig",
                "RetrieveStageConfig",
                "StageConfigRegistry",
                "TrainStageConfig",
                "build_stage_config_registry",
            },
        ),
        (
            "graph_memory.registry.training",
            {
                "ModelSelectionSettings",
                "RgcnMethodSettings",
                "RgcnModelSettings",
                "RgcnPairSamplingSettings",
                "RgcnTrainerSettings",
                "TrainDependencies",
                "TrainJobSettings",
                "TrainingRegistry",
                "TrainingReportingSettings",
                "build_training_registry",
            },
        ),
        (
            "scripts.workflow",
            {
                "build_stage_plan",
                "discover_ablation_variants",
                "format_commands",
                "format_status",
                "initialize_experiment",
                "inspect_experiment_status",
                "list_config_entries",
                "list_method_specs",
                "list_profile_specs",
                "list_recipe_specs",
                "list_stage_specs",
                "load_experiment_config",
                "load_manifest",
                "prune_manifest_completed_prefix",
                "run_stage_plan",
                "update_manifest_status",
            },
        ),
    ],
)
def test_public_modules_export_only_stable_entry_points(module_name: str, expected: set[str]) -> None:
    module = importlib.import_module(module_name)

    assert set(module.__all__) == expected


@pytest.mark.parametrize(
    ("module_name", "old_exports"),
    [
        ("graph_memory.contracts", {"JsonValue", "MemoryGraph", "RankedResult"}),
        ("graph_memory.evaluation", {"evaluate_results", "recall_at", "split_metric_tables"}),
        ("graph_memory.graphs", {"GraphBuilder", "GraphIndex", "build_graphs"}),
        ("graph_memory.graphs.construction", {"EdgeAccumulator", "GraphBuilder", "prepare_graph_input"}),
        ("graph_memory.graphs.construction.rules", {"BridgeEdgeRule", "GraphEdgeRule"}),
        (
            "graph_memory.models.graph_retriever.config",
            {"NodeFeatureConfig", "TrainableModelConfig", "default_model_config"},
        ),
        ("graph_memory.retrieval.execution", {"assemble_ranked_result", "run_retrieval"}),
        ("graph_memory.retrieval.methods.flat", {"BM25TaskRetriever", "DenseConfig", "ScorePipelineMethod"}),
        (
            "graph_memory.retrieval.methods.graph_rerank",
            {"GraphRerankConfig", "GraphRerankMethod", "normalize_scores", "rank_graph_from_initial_scores"},
        ),
        ("graph_memory.text", {"content_tokens", "extract_entities", "lexical_score"}),
    ],
)
def test_internal_packages_do_not_reexport_leaf_implementations(
    module_name: str,
    old_exports: set[str],
) -> None:
    module = importlib.import_module(module_name)

    assert module.__all__ == []
    assert old_exports.isdisjoint(vars(module))


def test_registry_projection_helpers_are_private() -> None:
    projections = importlib.import_module("graph_memory.registry.projections")
    retrieval = importlib.import_module("graph_memory.registry.retrieval")

    assert "legacy_builder_id_for" not in projections.__all__
    assert "project_retrieval_method_registry" not in projections.__all__
    assert not hasattr(projections, "legacy_builder_id_for")
    assert not hasattr(projections, "project_retrieval_method_registry")
    assert "require_payload" not in retrieval.__all__
    assert not hasattr(retrieval, "require_payload")

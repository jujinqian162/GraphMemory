import ast
import importlib
from pathlib import Path


MIGRATED_CONTRACT_NAMES = {
    "TaskId",
    "NodeId",
    "MethodName",
    "Score",
    "JsonObject",
    "JsonArray",
    "JsonValue",
    "NodeType",
    "EdgeType",
    "TrainPairSampleType",
    "ALLOWED_NODE_TYPES",
    "ALLOWED_EDGE_TYPES",
    "NEIGHBOR_TYPE_WEIGHT_EDGE_TYPES",
    "TRAIN_PAIR_SAMPLE_TYPES",
    "NEGATIVE_TRAIN_PAIR_SAMPLE_TYPES",
    "MemoryItem",
    "MemoryTaskInput",
    "MemoryTaskLabels",
    "CombinedMemoryTask",
    "QuestionNode",
    "GraphMemoryNode",
    "GraphNode",
    "GraphEdge",
    "MemoryGraph",
    "RankedNodeRecord",
    "RetrievedSubgraph",
    "RankedResult",
    "TrainPairRecord",
    "TrainPairBuildSummary",
    "MetricValue",
    "MetricRow",
    "MetricTableRow",
    "TaskMetricRow",
    "FailureCase",
    "GraphStatistics",
    "RunSummary",
}


def test_foundation_contract_modules_expose_artifact_contract_names():
    modules = {
        "graph_memory.contracts.common": [
            "TaskId",
            "NodeId",
            "MethodName",
            "Score",
            "JsonObject",
            "JsonArray",
            "JsonValue",
            "NodeType",
            "EdgeType",
            "TrainPairSampleType",
            "ALLOWED_NODE_TYPES",
            "ALLOWED_EDGE_TYPES",
            "NEIGHBOR_TYPE_WEIGHT_EDGE_TYPES",
            "TRAIN_PAIR_SAMPLE_TYPES",
            "NEGATIVE_TRAIN_PAIR_SAMPLE_TYPES",
        ],
        "graph_memory.contracts.tasks": [
            "MemoryItem",
            "MemoryTaskInput",
            "MemoryTaskLabels",
            "CombinedMemoryTask",
        ],
        "graph_memory.contracts.graphs": [
            "QuestionNode",
            "GraphMemoryNode",
            "GraphNode",
            "GraphEdge",
            "MemoryGraph",
        ],
        "graph_memory.contracts.ranking": [
            "RankedNodeRecord",
            "RetrievedSubgraph",
            "RankedResult",
        ],
        "graph_memory.contracts.training_pairs": [
            "TrainPairRecord",
            "TrainPairBuildSummary",
        ],
        "graph_memory.contracts.metrics": [
            "MetricValue",
            "MetricRow",
            "MetricTableRow",
            "TaskMetricRow",
            "FailureCase",
        ],
        "graph_memory.contracts.observability": [
            "GraphStatistics",
            "RunSummary",
        ],
    }

    for module_name, names in modules.items():
        module = importlib.import_module(module_name)
        for name in names:
            assert hasattr(module, name)


def test_validation_is_split_into_domain_modules_and_reexports_root_api():
    root_validation = importlib.import_module("graph_memory.validation")
    modules_by_function = {
        "graph_memory.validation.tasks": [
            "validate_no_label_fields",
            "validate_memory_task_inputs",
            "validate_memory_task_labels",
        ],
        "graph_memory.validation.graphs": ["validate_graphs"],
        "graph_memory.validation.ranking": ["validate_ranked_results"],
        "graph_memory.validation.training_pairs": [
            "validate_train_pairs",
            "validate_negative_sampling_config",
            "validate_train_pair_build_summary",
        ],
            "graph_memory.validation.model": [
                "validate_rgcn_model_config",
                "validate_rgcn_training_config",
                "validate_rgcn_checkpoint_metadata",
                "validate_graph_batch",
                "validate_training_batch",
                "validate_graph_rerank_config",
            ],
        "graph_memory.validation.metrics": ["validate_metric_rows"],
        "graph_memory.validation.common": ["ContractValidationError", "validate_task_id_alignment"],
    }

    for module_name, function_names in modules_by_function.items():
        module = importlib.import_module(module_name)
        for function_name in function_names:
            assert getattr(root_validation, function_name) is getattr(module, function_name)


def test_io_and_observability_root_ports_are_narrow_infrastructure_reexports():
    root_io = importlib.import_module("graph_memory.io")
    infra_io = importlib.import_module("graph_memory.infrastructure.io")
    runtime_environment = importlib.import_module("graph_memory.infrastructure.runtime_environment")
    run_summary = importlib.import_module("graph_memory.infrastructure.run_summary")
    root_observability = importlib.import_module("graph_memory.observability")

    io_names = {
        "read_json",
        "write_json",
        "read_csv",
        "write_csv",
        "write_jsonl",
        "merge_config",
    }
    assert set(root_io.__all__) == io_names
    for name in io_names:
        assert getattr(root_io, name) is getattr(infra_io, name)

    observability_names = {"now_iso", "collect_environment", "build_run_summary", "write_run_summary"}
    assert set(root_observability.__all__) == observability_names
    assert root_observability.collect_environment is runtime_environment.collect_environment
    assert root_observability.now_iso is run_summary.now_iso
    assert root_observability.build_run_summary is run_summary.build_run_summary
    assert root_observability.write_run_summary is run_summary.write_run_summary
    assert not hasattr(root_io, "load_config")
    for removed_name in ["graph_statistics", "config_digest", "build_score_debug_record"]:
        assert not hasattr(root_observability, removed_name)


def test_migrated_foundation_modules_do_not_depend_on_legacy_types_imports():
    checked_roots = [
        Path("graph_memory/contracts"),
        Path("graph_memory/infrastructure"),
        Path("graph_memory/validation"),
    ]
    for root in checked_roots:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "graph_memory.types":
                    raise AssertionError(f"{path} imports from graph_memory.types")
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "graph_memory.types":
                            raise AssertionError(f"{path} imports graph_memory.types")


def test_production_imports_use_contract_modules_for_migrated_artifacts():
    checked_roots = [Path("graph_memory"), Path("scripts")]
    violations: list[str] = []
    for root in checked_roots:
        for path in root.rglob("*.py"):
            if path == Path("graph_memory/types.py"):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "graph_memory.types":
                    imported = {alias.name for alias in node.names}
                    migrated = sorted(imported & MIGRATED_CONTRACT_NAMES)
                    if migrated:
                        violations.append(f"{path}: {migrated}")

    assert violations == []

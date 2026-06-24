from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

RETIRED_MODULES = {
    "graph_memory.config.training_compat",
    "graph_memory.training_config",
    "graph_memory.registry.projections",
    "graph_memory.retrieval_registry",
    "graph_memory.retrieval.catalog",
}

RETIRED_PATHS = {
    REPO_ROOT / "graph_memory" / "config" / "training_compat.py",
    REPO_ROOT / "graph_memory" / "training_config.py",
    REPO_ROOT / "graph_memory" / "registry" / "projections.py",
    REPO_ROOT / "graph_memory" / "retrieval_registry.py",
    REPO_ROOT / "graph_memory" / "retrieval" / "catalog.py",
    REPO_ROOT / "configs" / "training",
}

TRAINABLE_CONTRACT_PATHS = (
    REPO_ROOT / "configs" / "methods",
    REPO_ROOT / "graph_memory" / "models" / "dense_finetune",
    REPO_ROOT / "graph_memory" / "models" / "graph_retriever" / "checkpoint.py",
    REPO_ROOT / "graph_memory" / "registry",
    REPO_ROOT / "scripts" / "workflow",
    REPO_ROOT / "scripts" / "run_retrieval.py",
    REPO_ROOT / "scripts" / "train_method.py",
)

ACTIVE_DOC_PATHS = (
    REPO_ROOT / "docs" / "20-contracts",
    REPO_ROOT / "docs" / "configs",
    REPO_ROOT / "docs" / "40-operations",
)

FORBIDDEN_TOKENS = (
    "checkpoint_version",
    "effective_training_config",
    "normalize_raw_config",
    "builder_id",
    "_resolve_legacy_training_config",
    "_string_config_alias",
    "_json_bool_alias",
)

ACTIVE_DOC_FORBIDDEN_TOKENS = (
    "training_configs",
    "configs/training",
    "effective_training_config",
    "schema_version",
    "checkpoint_version",
    "builder_id",
    "requires_checkpoint",
    "requires_dense_encoder",
    "graph_memory/retrieval_registry.py",
    "graph_memory/retrieval/catalog.py",
)


def _is_importable(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def _contract_files() -> list[Path]:
    files: list[Path] = []
    for path in TRAINABLE_CONTRACT_PATHS:
        if path.is_file():
            files.append(path)
        elif path.exists():
            files.extend(
                child
                for child in path.rglob("*")
                if child.is_file() and child.suffix in {".py", ".json"}
            )
    return files


def _active_doc_files() -> list[Path]:
    files: list[Path] = []
    for path in ACTIVE_DOC_PATHS:
        if path.exists():
            files.extend(child for child in path.rglob("*.md") if child.is_file())
    return files


def test_retired_trainable_compatibility_modules_are_absent() -> None:
    existing_paths = sorted(str(path.relative_to(REPO_ROOT)) for path in RETIRED_PATHS if path.exists())
    importable_modules = sorted(module for module in RETIRED_MODULES if _is_importable(module))

    assert existing_paths == []
    assert importable_modules == []


def test_trainable_contracts_do_not_contain_versions_or_compatibility_tokens() -> None:
    offenders: list[str] = []
    for path in _contract_files():
        source = path.read_text(encoding="utf-8")
        relative = path.relative_to(REPO_ROOT)
        if "schema_version" in source and (
            "configs/methods" in relative.as_posix()
            or "dense_finetune" in relative.as_posix()
            or "scripts/workflow" in relative.as_posix()
        ):
            offenders.append(f"{relative}:schema_version")
        offenders.extend(f"{relative}:{token}" for token in FORBIDDEN_TOKENS if token in source)

    assert offenders == []


def test_experiment_configs_use_method_configs_only() -> None:
    offenders: list[str] = []
    for path in (REPO_ROOT / "configs" / "experiments").glob("*.json"):
        source = path.read_text(encoding="utf-8")
        if '"training_configs"' in source:
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == []


def test_active_docs_do_not_advertise_retired_trainable_surfaces() -> None:
    offenders: list[str] = []
    for path in _active_doc_files():
        source = path.read_text(encoding="utf-8")
        relative = path.relative_to(REPO_ROOT)
        offenders.extend(f"{relative}:{token}" for token in ACTIVE_DOC_FORBIDDEN_TOKENS if token in source)

    assert offenders == []


def test_canonical_trainable_method_configs_exist() -> None:
    config_root = REPO_ROOT / "configs" / "methods"

    assert (config_root / "dense_rgcn_graph_retriever.json").is_file()
    assert (config_root / "dense_ft_rgcn_graph_retriever.json").is_file()
    assert (config_root / "dense_ft.json").is_file()

"""Side-effecting input provisioning for the experiment flow.

These helpers run before the Prefect flow resolves any content-addressed
source. They only ensure that dataset raw files and encoder model directories
exist on disk; they never produce consumed artifacts, so downstream task cache
identity is unaffected. Existing paths are left untouched (existence test only).
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

from pydantic import BaseModel

from graph_memory.experiment.config import (
    DatasetName,
    DenseEncoderConfig,
    ResolvedExperimentConfig,
)

LOGGER = logging.getLogger("experiment.inputs")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# dataset config name -> (prepare_dataset registry key, data/<dir> directory).
_DATASET_REGISTRY_KEYS: dict[DatasetName, tuple[str, str]] = {
    "hotpotqa": ("hotpotqa-v1", "hotpotqa"),
    "twowiki": ("2wiki", "2wiki"),
    "twowiki_provenance": ("2wiki", "2wiki"),
    "musique": ("musique", "musique"),
}

# local encoder directory (repo-relative) -> Hugging Face repo id.
_MODEL_REPO_IDS: dict[str, str] = {
    "models/intfloat-e5-base-v2": "intfloat/e5-base-v2",
}


def ensure_inputs(
    config: ResolvedExperimentConfig,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> None:
    """Ensure dataset raw files and encoder models exist before the flow runs."""
    ensure_dataset(config, repository_root=repository_root)
    ensure_encoder_models(config, repository_root=repository_root)


def ensure_dataset(
    config: ResolvedExperimentConfig,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> None:
    """Download raw files for the configured dataset when any split is missing."""
    sources = [
        _resolve(split.source, repository_root)
        for split in config.dataset.splits.values()
    ]
    if all(source.exists() for source in sources):
        return
    entry = _DATASET_REGISTRY_KEYS.get(config.dataset.name)
    if entry is None:
        missing = ", ".join(str(s) for s in sources if not s.exists())
        raise FileNotFoundError(
            f"dataset {config.dataset.name!r} has no download entry; "
            f"place raw files manually. Missing: {missing}"
        )
    registry_key, directory = entry
    LOGGER.info(
        "dataset %s raw files missing; downloading via prepare_dataset (%s)",
        config.dataset.name,
        registry_key,
    )
    prepare_dataset = _load_prepare_dataset(repository_root)
    prepare_dataset.main(
        [
            "--dataset",
            registry_key,
            "--name",
            directory,
            "--data_dir",
            str(repository_root / "data"),
            "--no_verify",
        ]
    )


def ensure_encoder_models(
    config: ResolvedExperimentConfig,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> None:
    """Download local encoder model directories referenced by the config."""
    for model_name in sorted(_collect_model_names(config)):
        _ensure_model(model_name, repository_root=repository_root)


def _ensure_model(model_name: str, *, repository_root: Path) -> None:
    # Only local model paths are provisioned; name@revision is fetched on use.
    if "@" in model_name:
        return
    target = _resolve(Path(model_name), repository_root)
    if target.exists():
        return
    repo_id = _MODEL_REPO_IDS.get(model_name)
    if repo_id is None:
        raise FileNotFoundError(
            f"encoder model {model_name!r} is missing and has no download entry; "
            "place it manually or add a repo mapping in inputs.py"
        )
    from huggingface_hub import snapshot_download

    LOGGER.info("encoder model %s missing; downloading %s", model_name, repo_id)
    snapshot_download(repo_id=repo_id, local_dir=str(target))


def _collect_model_names(config: ResolvedExperimentConfig) -> set[str]:
    names: set[str] = set()
    _walk_for_models(config.method, names)
    transform = config.dataset.transform
    if transform is not None and transform.edge_scorer in {"dense", "hybrid"}:
        names.add(transform.dense_model)
    return names


def _walk_for_models(model: object, names: set[str]) -> None:
    if isinstance(model, DenseEncoderConfig):
        names.add(model.model_name)
        return
    if isinstance(model, BaseModel):
        for value in model.__dict__.values():
            _walk_for_models(value, names)
    elif isinstance(model, (list, tuple)):
        for value in model:
            _walk_for_models(value, names)


def _resolve(path: Path, repository_root: Path) -> Path:
    return path if path.is_absolute() else repository_root / path


def _load_prepare_dataset(repository_root: Path):
    module_path = repository_root / "scripts" / "prepare_dataset.py"
    spec = importlib.util.spec_from_file_location(
        "graph_memory_prepare_dataset", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load prepare_dataset from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


__all__ = [
    "ensure_dataset",
    "ensure_encoder_models",
    "ensure_inputs",
]

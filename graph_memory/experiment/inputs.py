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
    CrossEncoderMethodConfig,
    DatasetName,
    DenseEncoderConfig,
    ResolvedExperimentConfig,
)
from graph_memory.query_synthesis.provenance import authoring_metadata_path

LOGGER = logging.getLogger("experiment.inputs")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HUGGINGFACE_BASE_URL = "https://huggingface.co"
HUGGINGFACE_MIRROR_BASE_URL = "https://hf-mirror.com"

# dataset config name -> (prepare_dataset registry key, data/<dir> directory).
_DATASET_REGISTRY_KEYS: dict[DatasetName, tuple[str, str]] = {
    "hotpotqa": ("hotpotqa-v1", "hotpotqa"),
    "twowiki": ("2wiki", "2wiki"),
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
    if config.dataset.trajectory_source is not None:
        sources.append(_resolve(config.dataset.trajectory_source, repository_root))
    if config.dataset.natural_query_source is not None:
        sources.append(
            authoring_metadata_path(
                _resolve(config.dataset.natural_query_source, repository_root)
            )
        )
    entry = _DATASET_REGISTRY_KEYS.get(config.dataset.name)
    if all(source.exists() for source in sources):
        return
    if entry is None:
        missing = ", ".join(str(s) for s in sources if not s.exists())
        raise FileNotFoundError(
            f"dataset {config.dataset.name!r} has no download entry; "
            f"place raw files manually. Missing: {missing}"
        )
    registry_key, directory = entry
    prepare_dataset = _load_prepare_dataset(repository_root)
    LOGGER.info(
        "dataset %s raw files missing; downloading via prepare_dataset (%s)",
        config.dataset.name,
        registry_key,
    )
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
    try:
        snapshot_download(repo_id=repo_id, local_dir=str(target))
    except Exception:
        LOGGER.warning(
            "Hugging Face download failed for %s; retrying once via %s",
            repo_id,
            HUGGINGFACE_MIRROR_BASE_URL,
        )
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target),
            endpoint=HUGGINGFACE_MIRROR_BASE_URL,
        )


def _collect_model_names(config: ResolvedExperimentConfig) -> set[str]:
    names: set[str] = set()
    _walk_for_models(config.method, names)
    return names


def _walk_for_models(model: object, names: set[str]) -> None:
    if isinstance(model, CrossEncoderMethodConfig):
        names.add(model.backbone.model_name)
        return
    if isinstance(model, DenseEncoderConfig):
        names.add(model.model_name)
        return
    if isinstance(model, BaseModel):
        for value in model.__dict__.values():
            _walk_for_models(value, names)
    elif isinstance(model, (list, tuple)):
        for value in model:
            _walk_for_models(value, names)


def huggingface_mirror_url(url: str) -> str | None:
    """Return the equivalent mirror URL for an official Hugging Face URL."""
    if url == HUGGINGFACE_BASE_URL:
        return HUGGINGFACE_MIRROR_BASE_URL
    prefix = f"{HUGGINGFACE_BASE_URL}/"
    if not url.startswith(prefix):
        return None
    return f"{HUGGINGFACE_MIRROR_BASE_URL}/{url.removeprefix(prefix)}"


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
    "HUGGINGFACE_MIRROR_BASE_URL",
    "ensure_dataset",
    "ensure_encoder_models",
    "ensure_inputs",
    "huggingface_mirror_url",
]

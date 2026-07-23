from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from graph_memory.datasets.twowiki_provenance import (
    DenseRankerFactory,
    ProvenanceGraphConstructionConfig,
    TwoWikiProvenanceConversionResult,
    convert_twowiki_source_records,
    deterministic_dev_test_partition,
    resolve_worker_count,
)
from graph_memory.experiment.artifacts import (
    FileSourceRef,
    identify_external_source,
)
from graph_memory.experiment.config import (
    RankBucketConfig,
    TwoWikiProvenanceTransformConfig,
)
from graph_memory.io import read_json, write_json
from graph_memory.retrieval.contracts import SeedRanker

_SPLIT_FILES = ("train", "dev", "test")


@dataclass(frozen=True)
class TwoWikiProvenanceTransformResult:
    version_tag: str
    train: FileSourceRef
    dev: FileSourceRef
    test: FileSourceRef


def transform_version_tag(
    config: TwoWikiProvenanceTransformConfig,
    *,
    schema_version: int,
    encoder_digest: str | None = None,
) -> str:
    payload = {
        "schema_version": schema_version,
        "transform": config.identity(),
        "encoder_digest": encoder_digest,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    return f"v{schema_version}-{digest}"


def _graph_config(
    config: TwoWikiProvenanceTransformConfig,
) -> ProvenanceGraphConstructionConfig:
    return ProvenanceGraphConstructionConfig(
        strategy=config.edge_scorer,
        successors_per_output=config.successors_per_output,
        hybrid_dense_weight=config.hybrid_dense_weight,
        scorer_identity=config.scorer_identity,
        query_template_version=config.query_template_version,
        semantic_temperature=config.semantic_temperature,
        weight_floor=config.weight_floor,
        branch_policy_version=config.branch_policy_version,
        near_rank_bucket=_closed_bucket("near", config.near_rank_bucket),
        mid_rank_bucket=_closed_bucket("mid", config.mid_rank_bucket),
        tail_rank_bucket=(
            config.tail_rank_bucket.lower,
            config.tail_rank_bucket.upper,
        ),
    )


def _closed_bucket(name: str, bucket: RankBucketConfig) -> tuple[int, int]:
    if bucket.upper is None:
        raise ValueError(f"{name} rank bucket requires a finite upper bound")
    return (bucket.lower, bucket.upper)


def _dense_ranker(
    config: TwoWikiProvenanceTransformConfig,
    *,
    device: str,
) -> SeedRanker:
    from graph_memory.retrieval.methods.flat.dense import (
        DenseConfig,
        DenseTaskRetriever,
    )

    return DenseTaskRetriever(
        config=DenseConfig(
            model_name=config.dense_model,
            query_prefix=config.dense_query_prefix,
            passage_prefix=config.dense_passage_prefix,
            batch_size=config.dense_batch_size,
            device=device,
        ),
        device=device,
    )


def _dense_ranker_factory(
    config: TwoWikiProvenanceTransformConfig,
) -> DenseRankerFactory:
    return DenseRankerFactory(
        model_name=config.dense_model,
        query_prefix=config.dense_query_prefix,
        passage_prefix=config.dense_passage_prefix,
        batch_size=config.dense_batch_size,
    )


def _resolve_devices(
    config: TwoWikiProvenanceTransformConfig,
    *,
    is_dense: bool,
    fallback: str,
) -> tuple[str, ...]:
    if not is_dense:
        return ()
    if config.devices:
        return config.devices
    return (fallback,)


def _record_list(path: Path) -> list[object]:
    value = read_json(path)
    if not isinstance(value, list):
        raise ValueError(f"2Wiki source must be a JSON list: {path}")
    return list(cast("list[object]", value))


def _existing_result(
    version_dir: Path,
    *,
    version_tag: str,
    repository_root: Path,
) -> TwoWikiProvenanceTransformResult | None:
    files = {name: version_dir / f"{name}.json" for name in _SPLIT_FILES}
    if not all(path.is_file() for path in files.values()):
        return None
    refs = {
        name: _file_ref(path, repository_root=repository_root)
        for name, path in files.items()
    }
    return TwoWikiProvenanceTransformResult(version_tag=version_tag, **refs)


def _file_ref(path: Path, *, repository_root: Path) -> FileSourceRef:
    ref = identify_external_source(path, repository_root=repository_root)
    if not isinstance(ref, FileSourceRef):
        raise TypeError(f"transform split source must be a file: {ref.uri}")
    return ref


def materialize_transform_twowiki(
    *,
    train_source: FileSourceRef,
    dev_source: FileSourceRef,
    config: TwoWikiProvenanceTransformConfig,
    schema_version: int,
    output_root: Path,
    repository_root: Path,
    encoder_digest: str | None = None,
    device: str = "cpu",
) -> TwoWikiProvenanceTransformResult:
    version_tag = transform_version_tag(
        config, schema_version=schema_version, encoder_digest=encoder_digest
    )
    version_dir = output_root / version_tag
    existing = _existing_result(
        version_dir, version_tag=version_tag, repository_root=repository_root
    )
    if existing is not None:
        return existing

    graph_config = _graph_config(config)
    is_dense = config.edge_scorer in {"dense", "hybrid"}
    workers = resolve_worker_count(config.workers)

    # workers<=1 keeps the serial path (one main-process ranker); workers>1
    # ships a picklable factory so each worker builds its own ranker, since a
    # torch-backed ranker cannot cross a process boundary. `config.devices` is
    # the single source of GPU truth for both paths: the serial ranker binds to
    # the first configured card, parallel workers round-robin across all of them.
    if workers > 1:
        dense_ranker = None
        dense_ranker_factory = _dense_ranker_factory(config) if is_dense else None
        devices = _resolve_devices(config, is_dense=is_dense, fallback=device)
    else:
        serial_device = config.devices[0] if config.devices else device
        dense_ranker = (
            _dense_ranker(config, device=serial_device) if is_dense else None
        )
        dense_ranker_factory = None
        devices = ()

    def _convert(
        source: FileSourceRef, *, split: str
    ) -> TwoWikiProvenanceConversionResult:
        return convert_twowiki_source_records(
            _record_list(Path(source.uri)),
            candidate_cap=config.candidate_cap,
            seed=config.seed,
            strict=config.strict,
            graph_config=graph_config,
            dense_ranker=dense_ranker,
            workers=workers,
            dense_ranker_factory=dense_ranker_factory,
            devices=devices,
            progress_desc=f"transform twowiki provenance ({split})",
        )

    train_conversion = _convert(train_source, split="train")
    dev_conversion = _convert(dev_source, split="dev")
    dev_records, test_records = deterministic_dev_test_partition(
        dev_conversion.records,
        seed=config.seed,
        dev_fraction=config.dev_fraction,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    workspace = output_root / f".staging-{uuid.uuid4().hex}"
    workspace.mkdir(parents=True, exist_ok=False)
    try:
        write_json(workspace / "train.json", train_conversion.records)
        write_json(workspace / "dev.json", dev_records)
        write_json(workspace / "test.json", test_records)
        _publish(workspace, version_dir)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    result = _existing_result(
        version_dir, version_tag=version_tag, repository_root=repository_root
    )
    if result is None:
        raise RuntimeError(
            f"transform published but version dir is incomplete: {version_dir}"
        )
    return result


def _publish(workspace: Path, version_dir: Path) -> None:
    if version_dir.exists():
        return
    try:
        os.replace(workspace, version_dir)
    except OSError:
        if version_dir.exists():
            return
        raise


__all__ = [
    "TwoWikiProvenanceTransformResult",
    "materialize_transform_twowiki",
    "transform_version_tag",
]

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ImportanceAnnotationSettings:
    model_id: str
    model_path: Path
    prompt_version: str
    device: Literal["auto", "cuda", "cpu"]
    trust_remote_code: bool = True
    torch_dtype: str = "auto"
    low_cpu_mem_usage: bool = True
    tp_plan: None = None
    do_sample: Literal[False] = False
    use_cache: Literal[True] = True
    max_new_tokens: int = 2048

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("Importance annotation model_id must be non-empty.")
        if str(self.model_path) == "":
            raise ValueError("Importance annotation model_path must be non-empty.")
        if not self.prompt_version:
            raise ValueError("Importance annotation prompt_version must be non-empty.")
        if self.torch_dtype != "auto":
            raise ValueError("Importance annotation torch_dtype must be auto.")
        if self.tp_plan is not None:
            raise ValueError("Importance annotation tp_plan must be null.")
        if self.do_sample is not False:
            raise ValueError("Importance annotation do_sample must be false.")
        if self.use_cache is not True:
            raise ValueError("Importance annotation use_cache must be true.")
        if self.max_new_tokens <= 0:
            raise ValueError("Importance annotation max_new_tokens must be positive.")


__all__ = ["ImportanceAnnotationSettings"]

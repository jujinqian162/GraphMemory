from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class SentenceEncoder(Protocol):
    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        normalize_embeddings: bool = True,
    ) -> object:
        ...

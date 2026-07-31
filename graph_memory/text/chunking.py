from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast


class OffsetTokenizer(Protocol):
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
        truncation: bool,
    ) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class TokenChunk:
    text: str
    char_start: int
    char_end: int
    token_count: int


@dataclass(frozen=True)
class TokenChunkingConfig:
    tokenizer_name: str
    max_tokens: int = 512
    overlap_tokens: int = 64
    reserved_tokens: int = 0

    def __post_init__(self) -> None:
        if not self.tokenizer_name.strip():
            raise ValueError("tokenizer_name must be non-empty")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.reserved_tokens < 0:
            raise ValueError("reserved_tokens must be non-negative")
        if self.reserved_tokens >= self.max_tokens:
            raise ValueError("reserved_tokens must be smaller than max_tokens")
        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens must be non-negative")
        if self.overlap_tokens >= self.content_tokens:
            raise ValueError(
                "overlap_tokens must be smaller than the content token budget"
            )

    @property
    def content_tokens(self) -> int:
        return self.max_tokens - self.reserved_tokens


def load_offset_tokenizer(model_name_or_path: str | Path) -> OffsetTokenizer:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "transformers is required for token-aligned trajectory chunking"
        ) from error
    tokenizer = AutoTokenizer.from_pretrained(str(model_name_or_path), use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("trajectory chunking requires a fast tokenizer with offsets")
    return cast(OffsetTokenizer, cast(object, tokenizer))


def token_chunks(
    text: str,
    *,
    tokenizer: OffsetTokenizer,
    max_tokens: int,
    overlap_tokens: int,
) -> tuple[TokenChunk, ...]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must satisfy 0 <= overlap < max_tokens")
    if not text:
        return ()

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    raw_offsets = encoded.get("offset_mapping")
    if not isinstance(raw_offsets, Sequence):
        raise ValueError("tokenizer did not return offset_mapping")
    offsets: list[tuple[int, int]] = []
    for raw in raw_offsets:
        if (
            not isinstance(raw, Sequence)
            or isinstance(raw, (str, bytes))
            or len(raw) != 2
        ):
            raise ValueError("tokenizer returned an invalid offset mapping")
        start, end = raw
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("tokenizer offsets must be integers")
        if end > start:
            offsets.append((start, end))
    if not offsets:
        return (TokenChunk(text=text, char_start=0, char_end=len(text), token_count=0),)

    chunks: list[TokenChunk] = []
    step = max_tokens - overlap_tokens
    token_start = 0
    while token_start < len(offsets):
        token_end = min(token_start + max_tokens, len(offsets))
        char_start = 0 if token_start == 0 else offsets[token_start][0]
        char_end = len(text) if token_end == len(offsets) else offsets[token_end][0]
        if char_end <= char_start:
            char_start = offsets[token_start][0]
            char_end = offsets[token_end - 1][1]
        chunks.append(
            TokenChunk(
                text=text[char_start:char_end],
                char_start=char_start,
                char_end=char_end,
                token_count=token_end - token_start,
            )
        )
        if token_end == len(offsets):
            break
        token_start += step
    return tuple(chunks)


__all__ = [
    "OffsetTokenizer",
    "TokenChunk",
    "TokenChunkingConfig",
    "load_offset_tokenizer",
    "token_chunks",
]

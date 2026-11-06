"""Alignment of context characters with prompt tokens, from the tokenizer's offset mapping on the full prompt."""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Sequence

import numpy as np

Span = tuple[int, int]  # half-open character span


@dataclass(frozen=True)
class ContextTokens:
    """Location of the context in the tokenized prompt."""

    offsets: list[Span]   # character span of every prompt token
    char_start: int       # character offset of the context in the prompt
    start: int            # first context token
    end: int              # one past the last context token

    @property
    def n_tokens(self) -> int:
        return len(self.offsets)

    def content_span(self, token: int) -> Span:
        """Character span of a token relative to the context."""
        cs, ce = self.offsets[token]
        return cs - self.char_start, ce - self.char_start

    def is_zero_width(self, token: int) -> bool:
        cs, ce = self.offsets[token]
        return cs == ce


def locate_context(tokenizer, prompt: str, content: str) -> ContextTokens:
    """Context tokens: from the first token ending after the context start to the last token starting before its end."""
    offsets = [tuple(o) for o in tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)["offset_mapping"]]
    char_start = prompt.find(content)
    char_end = char_start + len(content)
    start = next(i for i, (_, ce) in enumerate(offsets) if ce > char_start)
    end = max(i for i, (cs, _) in enumerate(offsets) if cs < char_end) + 1
    return ContextTokens(offsets=offsets, char_start=char_start, start=start, end=end)


def token_segments(ctx: ContextTokens, segment_ends: Sequence[int], content_len: int) -> np.ndarray:
    """Segment index of every context token; -1 for zero-width tokens and tokens starting outside the context.

    A token belongs to the segment of its first character; whitespace between two segments belongs to the next one.
    """
    seg = np.full(ctx.end - ctx.start, -1, dtype=np.int64)
    for i in range(ctx.start, ctx.end):
        if ctx.is_zero_width(i):
            continue
        char_in_content, _ = ctx.content_span(i)
        if not 0 <= char_in_content < content_len:
            continue
        seg[i - ctx.start] = min(bisect.bisect_right(segment_ends, char_in_content), len(segment_ends) - 1)
    return seg


def merge_spans(spans: Sequence[Span]) -> list[Span]:
    """Sorted union of half-open spans."""
    merged: list[Span] = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def tokens_overlapping(ctx: ContextTokens, spans: Sequence[Span]) -> np.ndarray:
    """Mask of the context tokens that overlap any of the spans (character spans of the context)."""
    merged = merge_spans(spans)
    starts = [s for s, _ in merged]
    ends = [e for _, e in merged]
    mask = np.zeros(ctx.end - ctx.start, dtype=bool)
    for i in range(ctx.start, ctx.end):
        if ctx.is_zero_width(i):
            continue
        tok_start, tok_end = ctx.content_span(i)
        idx = bisect.bisect_right(ends, tok_start)
        mask[i - ctx.start] = idx < len(merged) and starts[idx] < tok_end
    return mask

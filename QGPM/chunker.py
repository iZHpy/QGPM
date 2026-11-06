"""Sentence-aligned segmentation of the context."""
from __future__ import annotations

import re

_ABBREV_RE = re.compile(r"\b(Mr|Mrs|Dr|Prof|Sr|Jr|vs|etc|i\.e|e\.g)\.")
_ABBREV_PLACEHOLDER = "\x00"


def _protect_abbrevs(text: str) -> str:
    """Replace the final '.' of abbreviations with a placeholder of the same length, so offsets are unchanged."""
    return _ABBREV_RE.sub(lambda m: m.group(0)[:-1] + _ABBREV_PLACEHOLDER, text)


def split_into_sentence_chunks(text: str, tokenizer, chunk_size: int) -> list[tuple[str, int, int]]:
    """Split text into chunks of whole sentences with at most `chunk_size` tokens each.

    A sentence longer than `chunk_size` forms a chunk of its own, so `chunk_size=1` gives one sentence per chunk.
    Returns `(chunk_text, char_start, char_end)` triples with offsets into `text`.
    """
    protected = _protect_abbrevs(text)
    sentence_spans: list[tuple[int, int]] = []
    prev = 0
    for m in re.finditer(r"(?<=[.!?])\s+", protected):
        if m.start() > prev:
            sentence_spans.append((prev, m.start()))
        prev = m.end()
    if prev < len(protected):
        sentence_spans.append((prev, len(protected)))

    chunks: list[tuple[str, int, int]] = []
    current: list[tuple[int, int]] = []
    n_current = 0
    for s, e in sentence_spans:
        sentence = text[s:e]
        if not sentence.strip():
            continue
        n = len(tokenizer.encode(sentence, add_special_tokens=False))
        if current and n_current + n > chunk_size:
            cs, ce = current[0][0], current[-1][1]
            chunks.append((text[cs:ce], cs, ce))
            current, n_current = [(s, e)], n
        else:
            current.append((s, e))
            n_current += n
    if current:
        cs, ce = current[0][0], current[-1][1]
        chunks.append((text[cs:ce], cs, ce))
    return chunks

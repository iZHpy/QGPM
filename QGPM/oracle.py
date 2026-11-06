"""Oracle scorer: one map in which the gold evidence gets score 1 and the rest of the context `low_score`.

The evidence comes from a dataset-specific locator that returns character spans of the item's content: the needle on
NoLiMa (with {CHAR} filled in), the evidence sentences on RULER qa, and each output value, expanded to its line or
sentence, on RULER niah.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import torch

from QGPM.region import apply_region
from QGPM.scorer import ScoringItem
from QGPM.token_map import Span, locate_context, tokens_overlapping

EvidenceLocator = Callable[[ScoringItem], list[Span]]


class NoLiMaNeedleLocator:
    """Needle span in a NoLiMa placement (q_id = "<test_name>_<depth>")."""

    def __init__(self, tests: Sequence[dict]) -> None:
        self._tests = {t["test_name"]: t for t in tests}

    def __call__(self, item: ScoringItem) -> list[Span]:
        test = self._tests[item.q_id.rsplit("_", 1)[0]]
        needle = test["needle"]
        candidates = [needle.replace("{CHAR}", c) for c in test["character_set"]] if "{CHAR}" in needle else [needle]
        spans = [(pos, pos + len(text)) for text in candidates if (pos := item.content.find(text)) >= 0]
        return spans[:1]


def _expand_to_sentence(haystack: str, lo: int, hi: int) -> Span:
    """Extend [lo, hi) to the enclosing line or sentence (bounded by "\\n" or ". ")."""
    start = max(haystack.rfind("\n", 0, lo) + 1, haystack.rfind(". ", 0, lo) + 2, 0)
    forward = [c for c in (haystack.find("\n", hi), haystack.find(". ", hi)) if c >= 0]
    end = min(forward) + 1 if forward else len(haystack)
    return start, max(end, hi)


def ruler_evidence_spans(record: dict) -> list[Span]:
    """Evidence spans in `record["haystack"]`: the `evidence` sentences (qa) or the `outputs` values (niah)."""
    haystack = record["haystack"]
    if record.get("evidence"):
        return [(pos, pos + len(s)) for s in record["evidence"] if s and (pos := haystack.find(s)) >= 0]
    spans = []
    for value in map(str, record["outputs"]):
        pos = haystack.find(value) if value else -1
        if pos >= 0:
            spans.append(_expand_to_sentence(haystack, pos, pos + len(value)))
    return spans


class RulerEvidenceLocator:
    """Evidence spans of RULER records (book_id = task, q_id = record position)."""

    def __init__(self, records_by_task: dict[str, list[dict]]) -> None:
        self._records = records_by_task

    def __call__(self, item: ScoringItem) -> list[Span]:
        return ruler_evidence_spans(self._records[item.book_id][int(item.q_id)])


class OracleScorer:
    """Single-map scorer: context tokens overlapping the evidence get 1, the other context tokens `low_score`."""

    def __init__(self, tokenizer, locate_evidence: EvidenceLocator, *, low_score: float, region: str) -> None:
        self.tokenizer = tokenizer
        self.locate_evidence = locate_evidence
        self.low_score = low_score
        self.region = region

    def score(self, item: ScoringItem, device: Any) -> torch.Tensor:
        ctx = locate_context(self.tokenizer, item.prompt, item.content)
        evidence = tokens_overlapping(ctx, self.locate_evidence(item))
        context_scores = np.where(evidence, 1.0, self.low_score)
        context_scores[[ctx.is_zero_width(i) for i in range(ctx.start, ctx.end)]] = 1.0
        scores = np.ones(ctx.n_tokens, dtype=np.float32)
        scores[ctx.start:ctx.end] = context_scores
        apply_region(scores, ctx, self.region, self.low_score)
        return torch.tensor(scores, device=device).unsqueeze(0)

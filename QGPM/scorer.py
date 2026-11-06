"""Scorer interface: the item a scorer receives and the per-token scores it returns."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch


@dataclass(frozen=True)
class ScoringItem:
    """One prompt to score; `content` appears verbatim in `prompt`."""

    book_id: str   # NoLiMa haystack or RULER task
    q_id: str      # item id within book_id
    query: str     # question the context is scored against
    content: str   # context to be scored
    prompt: str    # full chat prompt


class Scorer(Protocol):
    def score(self, item: ScoringItem, device: Any) -> torch.Tensor:
        """Scores in [0, 1] of shape [n_maps, n_prompt_tokens]; tokens outside the content get 1."""
        ...

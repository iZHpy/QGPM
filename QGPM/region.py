"""Region rule: the part of the context that a score map may compress."""
from __future__ import annotations

import numpy as np

from QGPM.token_map import ContextTokens

REGIONS = ("after", "all")


def apply_region(scores: np.ndarray, ctx: ContextTokens, region: str, low_score: float) -> None:
    """Apply the region rule to a score map in place.

    "after": context tokens before the first high-score token (score >= (low_score + 1) / 2) are set to 1, so only the
    text after the first evidence is compressed. "all": the map is left unchanged.
    """
    if region == "all":
        return
    context = scores[ctx.start:ctx.end]
    high = np.flatnonzero(context >= (low_score + 1.0) / 2)
    if high.size:
        context[: high[0]] = 1.0

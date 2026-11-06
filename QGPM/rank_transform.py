"""Rank transform of segment relevance into scores."""
from __future__ import annotations

import numpy as np


def rank_scores(relevance: np.ndarray, topk: int, T: float, low_score: float) -> np.ndarray:
    """Score in [low_score, 1] of every segment, from its relevance rank.

    f(rank) = sigmoid(((K - 0.5) - rank) / T), min-max normalized, and score = low_score + (1 - low_score) * f.
    T = 0 is a hard top-K: the K most relevant segments get 1, the others low_score.
    """
    k = min(max(topk, 1), len(relevance))
    ranks = (-relevance).argsort().argsort()  # 0 = most relevant
    if T == 0:
        return np.where(ranks < k, np.float32(1.0), np.float32(low_score)).astype(np.float32)
    logit = ((k - 0.5) - ranks.astype(np.float32)) / float(T)
    f = 1.0 / (1.0 + np.exp(-logit))
    f = (f - f.min()) / (f.max() - f.min())
    return (low_score + (1.0 - low_score) * f).astype(np.float32)

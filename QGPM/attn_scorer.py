"""Stage 1: segment relevance from the backbone's calibrated attention heads, shaped into score maps.

A plain prefill of the prompt captures, in the layers of the head set, the queries of the query rows (the last 16
prompt tokens and the question tokens) and the keys of all tokens, and stops after the deepest of these layers. For
every head h and context token j:

    logit_h(j) = max over rows r of  RoPE(q_h(r)) . RoPE(k_h(j)) / sqrt(d)

with RoPE at the original positions. Tokens whose text contains a newline are excluded. The logits are max-pooled per
segment (single sentences or sentence-aligned chunks), z-scored over the segments and averaged over the heads. Each
score map applies its own rank transform and the region rule to this relevance.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import torch

from QGPM.chunker import split_into_sentence_chunks
from QGPM.rank_transform import rank_scores
from QGPM.region import apply_region
from QGPM.score_cache import ScoreCache
from QGPM.scorer import ScoringItem
from QGPM.token_map import ContextTokens, locate_context, token_segments

N_LAST_ROWS = 16


@dataclass(frozen=True)
class HeadSet:
    """Ordered (layer, head) pairs, best first."""

    heads: tuple[tuple[int, int], ...]

    @classmethod
    def load(cls, path: str, n_heads: int) -> "HeadSet":
        """The first `n_heads` entries of a head ranking file (one `layer head` pair per line, `#` starts a comment)."""
        heads = []
        with open(path) as f:
            for line in f:
                fields = line.split("#", 1)[0].split()
                if fields:
                    heads.append((int(fields[0]), int(fields[1])))
        return cls(tuple(heads[:n_heads]))

    @property
    def layers(self) -> list[int]:
        return sorted({layer for layer, _ in self.heads})

    def digest(self) -> str:
        """Short hash of the head list, used in run and cache names."""
        return hashlib.sha1(",".join(f"{layer}.{head}" for layer, head in self.heads).encode()).hexdigest()[:8]


class _StopForward(Exception):
    """Raised after the deepest captured layer to end the forward pass early."""


def head_dim(config: Any) -> int:
    return getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads


class QKCapture:
    """Pre-RoPE queries (selected rows) and keys (all tokens) of some layers from one plain prefill.

    The hooks are registered only during `capture`, so generation with the same model is unaffected.
    """

    def __init__(self, model: Any, layers: Sequence[int]) -> None:
        self.model = model
        self.layers = sorted(set(layers))
        self.head_dim = head_dim(model.config)
        self._rows: Optional[torch.Tensor] = None
        self._store: dict[tuple[str, int], torch.Tensor] = {}

    def _register(self) -> list:
        handles = []
        for layer in self.layers:
            attn = self.model.model.layers[layer].self_attn
            q_module = getattr(attn, "q_norm", None) or attn.q_proj   # Qwen3: after the q/k norm
            k_module = getattr(attn, "k_norm", None) or attn.k_proj
            handles.append(q_module.register_forward_hook(self._q_hook(layer)))
            handles.append(k_module.register_forward_hook(self._k_hook(layer, layer == self.layers[-1])))
        return handles

    def _q_hook(self, layer: int):
        def hook(_module, _inputs, output):
            q = output[0].reshape(output.shape[1], -1, self.head_dim)          # [N, H, D]
            self._store[("q", layer)] = q[self._rows].transpose(0, 1).float()  # [H, R, D]
        return hook

    def _k_hook(self, layer: int, is_last: bool):
        def hook(_module, _inputs, output):
            k = output[0].reshape(output.shape[1], -1, self.head_dim)          # [N, KV, D]
            self._store[("k", layer)] = k.transpose(0, 1).contiguous()         # [KV, N, D], model dtype
            if is_last:
                raise _StopForward
        return hook

    @torch.no_grad()
    def capture(self, input_ids: torch.Tensor, rows: torch.Tensor) -> dict[tuple[str, int], torch.Tensor]:
        self._rows, self._store = rows, {}
        handles = self._register()
        try:
            self.model.model(input_ids=input_ids, use_cache=False)
        except _StopForward:
            pass
        finally:
            for handle in handles:
                handle.remove()
        store, self._store = self._store, {}
        return store


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """Standard RoPE in fp32 (HF rotate-half layout); `x` [..., n, D], `positions` [n]."""
    angles = positions.float()[:, None] * inv_freq.float()[None, :]           # [n, D/2]
    emb = torch.cat([angles, angles], dim=-1)                                  # [n, D]
    x = x.float()
    return x * emb.cos() + _rotate_half(x) * emb.sin()


def layer_head_logits(
    q: torch.Tensor,               # [H, R, D] fp32, pre-RoPE
    k: torch.Tensor,               # [KV, N, D], pre-RoPE
    heads: Sequence[int],
    row_positions: torch.Tensor,   # [R]
    inv_freq: torch.Tensor,        # [D/2]
) -> torch.Tensor:
    """Per head, the max over the query rows of the scaled q.k logits with RoPE: [len(heads), N]."""
    n_kv, n_tokens, d = k.shape
    group = q.shape[0] // n_kv
    head_idx = torch.tensor(heads, device=q.device)
    q = apply_rope(q, row_positions, inv_freq)
    k = apply_rope(k, torch.arange(n_tokens, device=k.device), inv_freq)
    return (torch.einsum("hrd,hnd->hrn", q[head_idx], k[head_idx // group]) / math.sqrt(d)).amax(dim=1)


def newline_token_ids(tokenizer) -> torch.Tensor:
    """Ids of the vocabulary entries whose decoded text contains a newline."""
    return torch.tensor([i for i in range(len(tokenizer)) if "\n" in tokenizer.decode([i])], dtype=torch.long)


def query_rows(item: ScoringItem, ctx: ContextTokens) -> list[int]:
    """The last N_LAST_ROWS prompt tokens and, when the question follows the context, the question tokens."""
    n = ctx.n_tokens
    rows = set(range(max(n - N_LAST_ROWS, 0), n))
    q_start = item.prompt.rfind(item.query)
    if q_start >= ctx.char_start + len(item.content):
        q_end = q_start + len(item.query)
        rows |= {i for i, (cs, ce) in enumerate(ctx.offsets) if cs < q_end and ce > q_start and cs != ce}
    return sorted(rows)


def segment_max(values: torch.Tensor, seg: torch.Tensor, n_segments: int) -> torch.Tensor:
    """Max of `values` [h, n] over the tokens of each segment (`seg` [n]); tokens with segment -1 or NaN are skipped.

    Returns [h, n_segments], with -inf for empty segments.
    """
    keep = (seg >= 0) & ~torch.isnan(values).any(dim=0)
    out = torch.full((values.shape[0], n_segments), float("-inf"), device=values.device)
    index = seg[keep].unsqueeze(0).expand(values.shape[0], -1)
    return out.scatter_reduce(1, index, values[:, keep].float(), reduce="amax", include_self=True)


def fuse_heads(segment_logits: np.ndarray) -> np.ndarray:
    """Z-score every head over the segments, then average over the heads: [h, n_seg] -> [n_seg] (-inf stays -inf)."""
    x = segment_logits.astype(np.float64)
    finite = np.isfinite(x)
    filled = np.where(finite, x, 0.0)
    per_head = np.maximum(finite.sum(axis=1, keepdims=True), 1)
    mean = filled.sum(axis=1, keepdims=True) / per_head
    std = np.sqrt((np.where(finite, x - mean, 0.0) ** 2).sum(axis=1, keepdims=True) / per_head) + 1e-6
    z = np.where(finite, (x - mean) / std, 0.0)
    per_segment = finite.sum(axis=0)
    fused = z.sum(axis=0) / np.maximum(per_segment, 1)
    return np.where(per_segment > 0, fused, -np.inf).astype(np.float32)


class AttnSignal:
    """Segment relevance of an item from the head set, computed once per item and optionally cached."""

    def __init__(self, model: Any, tokenizer, heads: HeadSet, cache: Optional[ScoreCache] = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.heads = heads
        self.cache = cache
        self.capture = QKCapture(model, heads.layers)
        self.inv_freq = model.model.rotary_emb.inv_freq
        self.newline_ids = newline_token_ids(tokenizer)

    def relevance(self, item: ScoringItem, ctx: ContextTokens, segment_ends: Sequence[int]) -> np.ndarray:
        """Relevance of every segment, float32 [n_segments]; higher means more relevant."""
        cached = self.cache.get(item.book_id, item.q_id) if self.cache is not None else None
        if cached is not None:
            return cached
        relevance = fuse_heads(self.segment_maxima(item, ctx, segment_ends))
        if self.cache is not None:
            self.cache.put(item.book_id, item.q_id, relevance)
        return relevance

    @torch.no_grad()
    def segment_maxima(self, item: ScoringItem, ctx: ContextTokens, segment_ends: Sequence[int]) -> np.ndarray:
        """Per head (in head-set order), the max logit within each segment, as float16 [n_heads, n_segments]."""
        device = next(self.model.parameters()).device
        ids = torch.tensor([self.tokenizer.encode(item.prompt, add_special_tokens=True)], device=device)
        rows = torch.tensor(query_rows(item, ctx), device=device)
        store = self.capture.capture(ids, rows)

        by_layer: dict[int, list[int]] = {}
        for layer, head in self.heads.heads:
            by_layer.setdefault(layer, []).append(head)
        per_layer, order = [], []
        for layer, heads in by_layer.items():
            logits = layer_head_logits(store[("q", layer)], store[("k", layer)], heads, rows, self.inv_freq.to(device))
            per_layer.append(logits[:, ctx.start:ctx.end])                      # [h_l, n_ctx]
            order += [(layer, h) for h in heads]
        values = torch.cat(per_layer)[[order.index(lh) for lh in self.heads.heads]]
        newline = torch.isin(ids[0, ctx.start:ctx.end], self.newline_ids.to(device))
        values[:, newline] = float("nan")
        seg = torch.tensor(token_segments(ctx, segment_ends, len(item.content)), device=device)
        return segment_max(values, seg, len(segment_ends)).cpu().numpy().astype(np.float16)


class AttnScorer:
    """QGPM scorer: the segment relevance of an item, shaped into one score map per `(K, T)` entry of `maps`.

    Each map gets its own rank transform and region rule; `score` returns [n_maps, n_prompt_tokens].
    """

    def __init__(self, signal: AttnSignal, *, segment_size: int, maps: Sequence[tuple[int, float]],
                 low_score: float, region: str) -> None:
        self.signal = signal
        self.segment_size = segment_size
        self.maps = list(maps)
        self.low_score = low_score
        self.region = region

    def score(self, item: ScoringItem, device: Any) -> torch.Tensor:
        tokenizer = self.signal.tokenizer
        ctx = locate_context(tokenizer, item.prompt, item.content)
        segment_ends = [end for _, _, end in split_into_sentence_chunks(item.content, tokenizer, self.segment_size)]
        relevance = self.signal.relevance(item, ctx, segment_ends)
        seg = token_segments(ctx, segment_ends, len(item.content))
        maps = [self._map(ctx, seg, rank_scores(relevance, topk, T, self.low_score)) for topk, T in self.maps]
        return torch.tensor(np.stack(maps), device=device)

    def _map(self, ctx: ContextTokens, seg: np.ndarray, segment_scores: np.ndarray) -> np.ndarray:
        scores = np.ones(ctx.n_tokens, dtype=np.float32)
        inside = seg >= 0
        scores[ctx.start:ctx.end][inside] = segment_scores[seg[inside]]
        apply_region(scores, ctx, self.region, self.low_score)
        return scores

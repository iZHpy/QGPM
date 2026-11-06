"""Stage 2: frequency-aware RoPE with one score map per decoder layer.

For token j and RoPE pair m (m = 0 rotates fastest):

    stride_m(j) = s(j) + (1 - s(j)) * sigmoid((m - m*) / tau_m),   clamped to [stride_min, 1]
    p'_{m,0} = 0,   p'_{m,j} = sum_{k=1..j} stride_m(k)

Low-relevance tokens advance less on the fast pairs (m < m*), while the slow pairs keep stride 1. Decoder layer l reads
map `layer_maps[l]` of the installed scores [n_maps, N]: a pre-hook on its self-attention replaces the position
embeddings passed in by the decoder layer. Generated tokens continue with stride 1 from each map's position of the last
prompt token, so new queries and cached keys share one coordinate system.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn


def m_star_from_anchor(anchor_length: float, head_dim: int, theta: float) -> float:
    """Fractional pair index whose wavelength is `anchor_length` (unscaled RoPE, omega_m = theta^(-2m/d))."""
    return head_dim * math.log(anchor_length / (2 * math.pi)) / (2 * math.log(theta))


def two_map_schedule(n_layers: int, split_layer: int) -> list[int]:
    """Map 0 in the layers below `split_layer`, map 1 from `split_layer` up."""
    return [0] * split_layer + [1] * (n_layers - split_layer)


def frequency_aware_rope(
    scores: torch.Tensor,     # [1, N]
    inv_freq: torch.Tensor,   # [D/2]
    m_star: float,
    tau_m: float,
    stride_min: float,
    out_dtype: torch.dtype,
    device_type: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prefill cos/sin [1, N, D] (rotate-half layout) and the per-pair position of the last token [1, D/2]."""
    with torch.autocast(device_type=device_type, enabled=False):
        inv_freq = inv_freq.float()
        s = scores.float().to(inv_freq.device).unsqueeze(-1)                          # [1, N, 1]
        m = torch.arange(inv_freq.shape[0], device=inv_freq.device, dtype=torch.float32)
        gate = torch.sigmoid((m - m_star) / tau_m).unsqueeze(0).unsqueeze(0)          # [1, 1, D/2]
        stride = (s + (1.0 - s) * gate).clamp(min=stride_min, max=1.0)                # [1, N, D/2]
        positions = torch.cumsum(stride, dim=1) - stride[:, :1, :]                    # p'_{m,0} = 0
        angles = positions * inv_freq.unsqueeze(0).unsqueeze(0)
        cos_half = torch.cos(angles).to(out_dtype)
        sin_half = torch.sin(angles).to(out_dtype)
        cos = torch.cat([cos_half, cos_half], dim=-1)
        sin = torch.cat([sin_half, sin_half], dim=-1)
        last = positions[:, -1, :]
    return cos, sin, last


class LayerMapRotary:
    """Stage 2 patch of a HF decoder: `set_scores` before a generation, `clear_scores` after it."""

    def __init__(self, model: nn.Module, layer_maps: Sequence[int], *, m_star: float, tau_m: float,
                 stride_min: float) -> None:
        self.layer_maps = list(layer_maps)
        self.m_star = float(m_star)
        self.tau_m = float(tau_m)
        self.stride_min = float(stride_min)
        self.inv_freq = model.model.rotary_emb.inv_freq
        self._scores: Optional[torch.Tensor] = None   # [n_maps, N]
        self._prefill: dict[int, tuple] = {}          # map -> (cos, sin, last position [1, D/2])
        self._decode: dict[tuple, tuple] = {}         # (map, first position id, n_tokens) -> (cos, sin)
        self._handles = [
            model.model.layers[layer].self_attn.register_forward_pre_hook(self._hook_for(g), with_kwargs=True)
            for layer, g in enumerate(self.layer_maps)
        ]

    def set_scores(self, scores: torch.Tensor) -> None:
        self._scores, self._prefill, self._decode = scores, {}, {}

    def clear_scores(self) -> None:
        self._scores, self._prefill, self._decode = None, {}, {}

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.clear_scores()

    def _hook_for(self, g: int):
        def hook(_module, args, kwargs):
            if self._scores is None:
                return None
            kwargs["position_embeddings"] = self._tables(g, kwargs)
            return args, kwargs
        return hook

    def _tables(self, g: int, kwargs: dict) -> tuple:
        like = kwargs["position_embeddings"][0]
        if kwargs["hidden_states"].shape[1] == self._scores.shape[1]:   # prefill of the whole prompt
            return self._prefill_tables(g, like)[:2]
        return self._decode_tables(g, kwargs["position_ids"], like)

    @torch.no_grad()
    def _prefill_tables(self, g: int, like: torch.Tensor) -> tuple:
        if g not in self._prefill:
            cos, sin, last = frequency_aware_rope(self._scores[g:g + 1], self.inv_freq, self.m_star, self.tau_m,
                                                  self.stride_min, out_dtype=like.dtype, device_type=like.device.type)
            self._prefill[g] = (cos.to(like.device), sin.to(like.device), last)
        return self._prefill[g]

    @torch.no_grad()
    def _decode_tables(self, g: int, position_ids: torch.Tensor, like: torch.Tensor) -> tuple:
        key = (g, int(position_ids.reshape(-1)[0]), position_ids.shape[-1])
        if key not in self._decode:
            anchor = self._prefill[g][2]                                                        # [1, D/2]
            with torch.autocast(device_type=like.device.type, enabled=False):
                inv_freq = self.inv_freq.float()
                offset = (position_ids - (self._scores.shape[1] - 1)).float().to(inv_freq.device)  # [B, S]
                positions = anchor.float().to(inv_freq.device)[:, None, :] + offset[..., None]     # [B, S, D/2]
                angles = positions * inv_freq
                cos = torch.cat([angles.cos(), angles.cos()], dim=-1).to(like.dtype)
                sin = torch.cat([angles.sin(), angles.sin()], dim=-1).to(like.dtype)
            self._decode[key] = (cos.to(like.device), sin.to(like.device))
        return self._decode[key]

"""Head ranking for the QGPM scorer from the signals of tools/collect_attn_signals.py (CPU).

Each head is ranked by the mean reciprocal rank of the best needle chunk over the items, with ties counted as half
(#higher + #tied / 2). The heads are written best first, one `layer head  # mrr` line each, after a header line.

    python3 tools/calibrate_heads.py --signals signals/<model> --out heads/<model>.txt
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch


def load_items(directory: str) -> list[dict]:
    items = []
    for path in sorted(glob.glob(os.path.join(directory, "shard*.pt"))):
        items += torch.load(path, weights_only=False)
    return [it for it in items if it["gold"]]


def expected_rank(scores: np.ndarray, gold: list[int]) -> np.ndarray:
    """0-based rank of the best gold chunk over the last axis of `scores`, with random tie-breaking."""
    best_gold = scores[..., gold].max(-1, keepdims=True)
    higher = (scores > best_gold).sum(-1)
    tied = (scores == best_gold).sum(-1) - 1
    return higher + tied / 2.0


def per_head_mrr(items: list[dict]) -> np.ndarray:
    """Mean reciprocal rank of the gold chunk for every head, [L, H]."""
    total = None
    for it in items:
        rr = 1.0 / (1.0 + expected_rank(it["signals"].float().numpy(), it["gold"]))
        total = rr if total is None else total + rr
    return total / len(items)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--signals", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    with open(os.path.join(args.signals, "meta.json")) as f:
        meta = json.load(f)
    items = load_items(args.signals)
    mrr = per_head_mrr(items)
    heads = [np.unravel_index(i, mrr.shape) for i in np.argsort(-mrr, axis=None)]
    model = meta["model"].rstrip("/").split("/")[-1]
    every = {1: "", 2: "2nd ", 3: "3rd "}.get(meta["every"], f"{meta['every']}th ")
    with open(args.out, "w") as f:
        f.write(f"# {model}: heads ranked by the mean reciprocal rank of the needle chunk on NoLiMa "
                f"{meta['context_length'] // 1000}K ({' + '.join(meta['books'])}, every {every}placement, "
                f"n = {len(items)})\n")
        for layer, head in heads:
            f.write(f"{layer} {head}  # {mrr[layer, head]:.4f}\n")
    print("top-8: " + ", ".join(f"({layer},{head}) {mrr[layer, head]:.3f}" for layer, head in heads[:8]))


if __name__ == "__main__":
    main()

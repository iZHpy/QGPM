"""Per-head attention signals on NoLiMa placements, the input of tools/calibrate_heads.py.

For every selected placement, one plain prefill of the evaluation prompt captures the queries and keys of all layers.
Per head, the RoPE logits from the query rows (the last 16 prompt tokens and the question tokens) to the context tokens,
newline tokens excluded, are max-pooled over 64-token sentence chunks. Writes `shard*.pt` (one dict per item:
`signals` [L, H, n_chunks] fp16, `gold` = ids of the chunks overlapping the needle) and `meta.json`.

    CUDA_VISIBLE_DEVICES=0 python3 tools/collect_attn_signals.py --model <model> --out signals/<model>
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM

from config import nolima_length_tag
from pipeline.data import NoLiMaPlacements, load_nolima_tests, nolima_depths, save_json
from pipeline.prompts import nolima_prompt
from QGPM.attn_scorer import QKCapture, apply_rope, newline_token_ids, query_rows, segment_max
from QGPM.chunker import split_into_sentence_chunks
from QGPM.oracle import NoLiMaNeedleLocator
from QGPM.scorer import ScoringItem
from QGPM.token_map import locate_context, token_segments
from runners import load_tokenizer

NEEDLE_SET = "Datasets/NoLiMa/data/needlesets/needle_set.json"
HAYSTACK_DIR = "Datasets/NoLiMa/data/haystack/rand_shuffle"
CHUNK_SIZE = 64
SHARD = 250


def nolima_items(args, tokenizer):
    """`(item, needle spans, meta)` of every `--every`-th placement of each book, in test and depth order."""
    tests = load_nolima_tests(NEEDLE_SET)
    locator = NoLiMaNeedleLocator(tests)
    cache_dir = f"Log_placements/NoLiMa_{nolima_length_tag(args.context_length)}_needle_set_0.0_100.0_26_shift0_seed42"
    for index, haystack in enumerate(sorted(os.listdir(HAYSTACK_DIR))):
        placements = NoLiMaPlacements(os.path.join(HAYSTACK_DIR, haystack), index, cache_dir,
                                      context_length=args.context_length, depths=nolima_depths(0.0, 100.0, 26),
                                      shift=0, base_seed=42)
        if placements.name not in args.books:
            continue
        book = ((test, p) for test in tests for p in placements.for_test(test)[:: args.depth_step])
        for i, (test, p) in enumerate(book):
            if i % args.every:
                continue
            prompt = nolima_prompt(tokenizer, test["task_template"], p["text"], p["question"])
            item = ScoringItem(placements.name, f"{test['test_name']}_{round(p['depth'], 4)}", p["question"],
                               p["text"], prompt)
            yield item, locator(item), {"depth": p["depth"], "test_name": test["test_name"]}


@torch.no_grad()
def item_signals(model, tokenizer, capture: QKCapture, item: ScoringItem, newline_ids: torch.Tensor) -> tuple:
    device = next(model.parameters()).device
    ctx = locate_context(tokenizer, item.prompt, item.content)
    ids = torch.tensor([tokenizer.encode(item.prompt, add_special_tokens=True)], device=device)
    rows = torch.tensor(query_rows(item, ctx), device=device)
    store = capture.capture(ids, rows)

    chunks = split_into_sentence_chunks(item.content, tokenizer, CHUNK_SIZE)
    seg = torch.tensor(token_segments(ctx, [c[2] for c in chunks], len(item.content)), device=device)
    newline = torch.isin(ids[0, ctx.start:ctx.end], newline_ids)
    inv_freq = model.model.rotary_emb.inv_freq.to(device)
    positions = torch.arange(ids.shape[1], device=device)
    per_layer = []
    for layer in range(model.config.num_hidden_layers):
        q, k = store[("q", layer)], store[("k", layer)]                          # [H, R, D], [KV, N, D]
        group = q.shape[0] // k.shape[0]
        qq = apply_rope(q, rows, inv_freq)
        kk = apply_rope(k, positions, inv_freq)[:, ctx.start:ctx.end].repeat_interleave(group, dim=0)
        logits = (torch.einsum("hrd,hnd->hrn", qq, kk) / q.shape[-1] ** 0.5).amax(1)   # [H, n_ctx]
        logits[:, newline] = float("nan")
        per_layer.append(segment_max(logits, seg, len(chunks)).half().cpu())
    return torch.stack(per_layer), chunks                                          # [L, H, n_chunks]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--context_length", type=int, default=8000)
    p.add_argument("--books", nargs="+", default=["rand_book_1", "rand_book_2"])
    p.add_argument("--every", type=int, default=2, help="keep every k-th placement")
    p.add_argument("--depth_step", type=int, default=1, help="every k-th depth of the 26-point grid")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tokenizer = load_tokenizer(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": 0},
                                                 attn_implementation="flash_attention_2").eval()
    capture = QKCapture(model, range(model.config.num_hidden_layers))
    newline_ids = newline_token_ids(tokenizer).to(model.device)

    shard, n_done, t0 = [], 0, time.time()
    for item, spans, meta in nolima_items(args, tokenizer):
        signals, chunks = item_signals(model, tokenizer, capture, item, newline_ids)
        gold = [i for i, (_, cs, ce) in enumerate(chunks) if any(s < ce and cs < e for s, e in spans)]
        shard.append({"book_id": item.book_id, "q_id": item.q_id, "n_chunks": len(chunks), "gold": gold,
                      "signals": signals, **meta})
        n_done += 1
        if len(shard) == SHARD:
            torch.save(shard, os.path.join(args.out, f"shard{n_done // SHARD:03d}.pt"))
            shard = []
        if n_done % 50 == 0:
            print(f"{n_done} items, {time.time() - t0:.0f}s", flush=True)
    if shard:
        torch.save(shard, os.path.join(args.out, f"shard{n_done // SHARD + 1:03d}.pt"))
    save_json(os.path.join(args.out, "meta.json"), {"model": args.model, "context_length": args.context_length,
                                                     "books": args.books, "every": args.every,
                                                     "depth_step": args.depth_step, "n_items": n_done})
    print(f"DONE {n_done} items in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""Stage 2 checks for a backbone on three NoLiMa 8K placements (GPU, about 10 minutes on one A100).

    CUDA_VISIBLE_DEVICES=0 python3 tools/check_identity.py --model mistralai/Ministral-8B-Instruct-2410

0. The prompt starts with at most one BOS, and the Base answers are not empty.
1. All-ones scores reproduce Base exactly (prefill logits and greedy tokens), with one map and with two maps.
2. Two equal maps reproduce one map in every layer exactly.
3. A different upper map changes the output.
4. Greedy decoding with the KV cache (positions p'_{N-1} + k) matches one prefill of prompt + generated tokens, with
   the scores extended by 1, about as closely as plain decoding matches its own prefill; a step whose greedy token
   differs must be a near-tie. As a control, decoding at the plain positions N - 1 + k must deviate clearly more.

Settings: anchor 12000, tau_m 3, stride_min 0.05; Oracle scores (needle 1, rest 0.05, region all).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM

from pipeline.data import NoLiMaPlacements, load_nolima_tests, nolima_depths
from pipeline.prompts import nolima_prompt
from QGPM.oracle import NoLiMaNeedleLocator, OracleScorer
from QGPM.rope import LayerMapRotary, m_star_from_anchor, two_map_schedule
from QGPM.scorer import ScoringItem
from runners import load_tokenizer

NEEDLE_SET = "Datasets/NoLiMa/data/needlesets/needle_set.json"
HAYSTACK = "Datasets/NoLiMa/data/haystack/rand_shuffle/rand_book_3.txt"
PLACEMENTS = "Log_placements/NoLiMa_8K_needle_set_0.0_100.0_26_shift0_seed42"
ITEMS = (("0401_T17_C02_onehop", 12), ("0402_T01_C02_twohop", 4), ("0405Inv_T01_C02_onehop", 24))  # (test, depth index)
ANCHOR, TAU_M, STRIDE_MIN, LOW_SCORE = 12000, 3.0, 0.05, 0.05
DECODE_STEPS = 8
BROAD_SPAN = 500        # context tokens set to 1 in the upper test map, so its strides and anchor differ
DEVICE = "cuda:0"


class PlainDecodeRotary(LayerMapRotary):
    """Control for check 4: generated tokens at the plain positions N - 1 + k instead of p'_{N-1} + k."""

    def _tables(self, g: int, kwargs: dict) -> tuple:
        if kwargs["hidden_states"].shape[1] == self._scores.shape[1]:
            return super()._tables(g, kwargs)
        return kwargs["position_embeddings"]


def load_items(tokenizer, tests: dict) -> list[tuple[torch.Tensor, ScoringItem]]:
    placements = NoLiMaPlacements(HAYSTACK, 2, PLACEMENTS, context_length=8000, depths=nolima_depths(0.0, 100.0, 26),
                                  shift=0, base_seed=42)
    out = []
    for test_name, depth_index in ITEMS:
        test = tests[test_name]
        p = placements.for_test(test)[depth_index]
        prompt = nolima_prompt(tokenizer, test["task_template"], p["text"], p["question"])
        item = ScoringItem("rand_book_3", f"{test_name}_{round(p['depth'], 4)}", p["question"], p["text"], prompt)
        out.append((tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(DEVICE), item))
    return out


@torch.no_grad()
def run(model, ids, patcher=None, scores=None, new_tokens: int = 12) -> tuple[torch.Tensor, list[int]]:
    """Last-position prefill logits and the greedy continuation."""
    if patcher is not None:
        patcher.set_scores(scores)
    try:
        logits = model(input_ids=ids).logits[0, -1].float()
        gen = model.generate(input_ids=ids, max_new_tokens=new_tokens, do_sample=False, use_cache=True)[0, ids.shape[1]:]
    finally:
        if patcher is not None:
            patcher.clear_scores()
    return logits, gen.tolist()


@torch.no_grad()
def decode_vs_prefill(model, ids, patcher=None, scores=None) -> tuple[float, float]:
    """Max |delta logit| between the greedy decode steps and one prefill of prompt + generated tokens, and the smallest
    top-2 margin of a step whose greedy token differs (`inf` when all steps agree)."""
    if patcher is not None:
        patcher.set_scores(scores)
    try:
        out = model.generate(input_ids=ids, max_new_tokens=DECODE_STEPS, do_sample=False, use_cache=True,
                             output_logits=True, return_dict_in_generate=True)
    finally:
        if patcher is not None:
            patcher.clear_scores()
    generated = out.sequences[:, ids.shape[1]:]
    decoded = torch.stack(out.logits, dim=1)[0].float()                                   # [k, V]
    full = torch.cat([ids, generated[:, :-1]], dim=1)
    if patcher is not None:
        extension = torch.ones(scores.shape[0], generated.shape[1] - 1, device=scores.device)
        patcher.set_scores(torch.cat([scores, extension], dim=1))                        # generated tokens: stride 1
    try:
        prefilled = model(input_ids=full).logits[0, ids.shape[1] - 1:].float()           # [k, V]
    finally:
        if patcher is not None:
            patcher.clear_scores()
    differs = decoded.argmax(-1) != prefilled.argmax(-1)
    top2 = prefilled.topk(2, dim=-1).values
    margin = (top2[:, 0] - top2[:, 1])[differs]
    return (decoded - prefilled).abs().max().item(), (margin.min().item() if margin.numel() else float("inf"))


def broad_map(oracle_map: torch.Tensor) -> torch.Tensor:
    out = oracle_map.clone()
    start = int((oracle_map[0] < 1).nonzero()[0])
    out[0, start:start + BROAD_SPAN] = 1.0
    return out


def check_prompts(model, tokenizer, items) -> None:
    bos = tokenizer.bos_token_id
    for ids, item in items:
        first = ids[0, :3].tolist()
        n_bos = int(first[0] == bos) + int(first[1] == bos) if bos is not None else 0
        assert n_bos <= 1, f"check 0: {n_bos} BOS tokens at the prompt start {first}"
        gen = model.generate(input_ids=ids, max_new_tokens=32, do_sample=False, use_cache=True)[0, ids.shape[1]:]
        answer = tokenizer.decode(gen, skip_special_tokens=True).strip()
        print(f"check 0 {item.q_id}: {ids.shape[1]} tokens, first ids {first} ({n_bos} BOS); Base answer {answer!r}, "
              f"stopped at EOS: {tokenizer.eos_token_id in gen.tolist()}")
        assert answer, "check 0: empty Base answer"
    print("check 0 OK (prompt: at most one BOS; non-empty Base answers)")


def check_all_ones(model, items, base, schedules, gate) -> None:
    for name, schedule in schedules.items():
        patcher = LayerMapRotary(model, schedule, **gate)
        for (ids, item), (lb, gb) in zip(items, base):
            n_maps = max(schedule) + 1
            lo, go = run(model, ids, patcher, torch.ones(n_maps, ids.shape[1], device=DEVICE))
            assert torch.equal(lo, lb) and go == gb, f"check 1: all-ones ({name}) != Base on {item.q_id}"
        patcher.remove()
    print("check 1 OK (all-ones scores reproduce Base exactly, with one map and with two maps)")


def check_equal_maps(model, items, single, oracle_maps, schedules, gate) -> None:
    patcher = LayerMapRotary(model, schedules["two maps"], **gate)
    for (ids, item), (ls, gs), scores in zip(items, single, oracle_maps):
        lo, go = run(model, ids, patcher, torch.cat([scores, scores], 0))
        assert torch.equal(lo, ls) and go == gs, f"check 2: two equal maps != one map on {item.q_id}"
    patcher.remove()
    print("check 2 OK (two equal maps reproduce one map in every layer exactly)")


def check_two_maps_differ(model, items, single, oracle_maps, schedules, gate) -> None:
    ids, _ = items[0]
    patcher = LayerMapRotary(model, schedules["two maps"], **gate)
    lt, _ = run(model, ids, patcher, torch.cat([oracle_maps[0], broad_map(oracle_maps[0])], 0))
    patcher.remove()
    assert not torch.equal(lt, single[0][0]), "check 3: a different upper map did not change the output"
    print(f"check 3 OK (a different upper map changes the output: max |delta logit| "
          f"{(lt - single[0][0]).abs().max().item():.3f})")


def _consistent(result: tuple[float, float], tolerance: float) -> bool:
    """Logits within the tolerance, and every differing greedy step a near-tie within it."""
    delta, min_margin = result
    return delta <= tolerance and (min_margin == float("inf") or min_margin <= tolerance)


def _fmt(result: tuple[float, float]) -> str:
    delta, min_margin = result
    return f"{delta:.3f} ({'same tokens' if min_margin == float('inf') else f'tie margin {min_margin:.3f}'})"


def check_decode_coordinates(model, items, oracle_maps, schedules, gate) -> None:
    for (ids, item), oracle_map in zip(items, oracle_maps):
        plain = decode_vs_prefill(model, ids)
        one = LayerMapRotary(model, schedules["one map"], **gate)
        single = decode_vs_prefill(model, ids, one, oracle_map)
        one.remove()
        two = LayerMapRotary(model, schedules["two maps"], **gate)
        layered = decode_vs_prefill(model, ids, two, torch.cat([oracle_map, broad_map(oracle_map)], 0))
        two.remove()
        bug = PlainDecodeRotary(model, schedules["one map"], **gate)
        control = decode_vs_prefill(model, ids, bug, oracle_map)
        bug.remove()
        print(f"check 4 {item.q_id}: max |delta logit| decode vs prefill: plain {_fmt(plain)}, one map {_fmt(single)}, "
              f"two maps {_fmt(layered)}, plain-position control {_fmt(control)}")
        tolerance = max(3 * plain[0], 0.5)
        assert _consistent(single, tolerance), "check 4: one-map decode coordinates"
        assert _consistent(layered, tolerance), "check 4: two-map decode coordinates"
        assert control[0] > 2 * max(single[0], layered[0]), "check 4: the power control does not separate"
    print("check 4 OK (decode coordinates consistent with one and two maps; the control separates)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
                                                 device_map={"": 0}).eval()
    n_layers = model.config.num_hidden_layers
    half = n_layers // 2
    m_star = m_star_from_anchor(ANCHOR, 2 * model.model.rotary_emb.inv_freq.numel(), model.config.rope_parameters["rope_theta"])
    gate = dict(m_star=m_star, tau_m=TAU_M, stride_min=STRIDE_MIN)
    schedules = {"one map": [0] * n_layers, "two maps": two_map_schedule(n_layers, half)}
    tests = {t["test_name"]: t for t in load_nolima_tests(NEEDLE_SET)}
    oracle = OracleScorer(tokenizer, NoLiMaNeedleLocator(list(tests.values())), low_score=LOW_SCORE, region="all")
    items = load_items(tokenizer, tests)
    print(f"{args.model}: {n_layers} layers, split at {half}; m* = {m_star:.4f} at anchor {ANCHOR}; "
          f"prompt tokens {[ids.shape[1] for ids, _ in items]}")

    check_prompts(model, tokenizer, items)
    oracle_maps = [oracle.score(item, DEVICE) for _, item in items]
    base = [run(model, ids) for ids, _ in items]
    one = LayerMapRotary(model, schedules["one map"], **gate)
    single = [run(model, ids, one, scores) for (ids, _), scores in zip(items, oracle_maps)]
    one.remove()
    check_all_ones(model, items, base, schedules, gate)
    check_equal_maps(model, items, single, oracle_maps, schedules, gate)
    check_two_maps_differ(model, items, single, oracle_maps, schedules, gate)
    check_decode_coordinates(model, items, oracle_maps, schedules, gate)
    assert sum(len(m._forward_pre_hooks) for m in model.modules()) == 0, "pre-hooks left on the model"
    for (_, item), (_, gs), (_, gb) in zip(items, single, base):
        print(f"{item.q_id} | Base: {tokenizer.decode(gb)!r} | Oracle: {tokenizer.decode(gs)!r}")
    print("OK: identity checks passed")


if __name__ == "__main__":
    main()

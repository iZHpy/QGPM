"""Model loading, method components, and the NoLiMa and RULER run loops.

Base runs plain RoPE. QGPM installs two attention score maps per item (the lower map below the split layer, the upper
map from the split layer up); Oracle installs one gold-evidence map in every layer.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Callable

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from pipeline.data import NoLiMaPlacements, load_nolima_tests, load_ruler_records, nolima_depths, save_json
from pipeline.evaluate import NOLIMA_METRICS, NoLiMaEvaluator, RULEREvaluator
from pipeline.generate import GenerationConfig
from QGPM.attn_scorer import AttnScorer, AttnSignal, HeadSet
from QGPM.oracle import NoLiMaNeedleLocator, OracleScorer, RulerEvidenceLocator
from QGPM.rope import LayerMapRotary, m_star_from_anchor, two_map_schedule
from QGPM.score_cache import ScoreCache


def load_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(args: argparse.Namespace):
    dtype = getattr(torch, args.dtype)
    kwargs: dict[str, Any] = dict(dtype=dtype, device_map=args.device_map)
    if args.load_in_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True,
                                                           llm_int8_skip_modules=["lm_head", "embed_tokens"])
    elif args.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
    if not args.no_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"
    return AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **kwargs).eval()


def build_method(args: argparse.Namespace, model: Any, tokenizer, locate_evidence: Callable[[], Any]) -> tuple:
    """Return `(rope_patcher, scorer, config)` of the run's method, or `(None, None, {})` for Base."""
    if args.method == "Base":
        return None, None, {}
    n_layers = model.config.num_hidden_layers
    m_star = args.m_star if args.m_star is not None else m_star_from_anchor(
        args.anchor_length, 2 * model.model.rotary_emb.inv_freq.numel(), model.config.rope_parameters["rope_theta"])
    stage2 = dict(m_star=m_star, tau_m=args.tau_m, stride_min=args.stride_min)
    config = {"m_star": m_star, "anchor_length": args.anchor_length, "tau_m": args.tau_m,
              "stride_min": args.stride_min, "low_score": args.low_score, "region": args.region}
    if args.method == "Oracle":
        scorer = OracleScorer(tokenizer, locate_evidence(), low_score=args.low_score, region=args.region)
        return LayerMapRotary(model, [0] * n_layers, **stage2), scorer, config

    split = args.split_layer if args.split_layer is not None else n_layers // 2
    heads = HeadSet.load(args.heads, args.n_heads)
    cache = ScoreCache(args.score_cache_path) if args.score_cache_path else None
    scorer = AttnScorer(AttnSignal(model, tokenizer, heads, cache),
                        segment_size=1 if args.segment == "sent" else args.chunk_size,
                        maps=[(args.lower_topk, args.lower_T), (args.upper_topk, args.upper_T)],
                        low_score=args.low_score, region=args.region)
    config.update(heads=args.heads, n_heads=args.n_heads, head_list=[list(h) for h in heads.heads],
                  segment=args.segment, chunk_size=args.chunk_size, lower_topk=args.lower_topk, lower_T=args.lower_T,
                  upper_topk=args.upper_topk, upper_T=args.upper_T, split_layer=split,
                  score_cache_path=args.score_cache_path)
    return LayerMapRotary(model, two_map_schedule(n_layers, split), **stage2), scorer, config


def _setup(args: argparse.Namespace, evaluator_cls: type, locate_evidence: Callable[[], Any]) -> tuple:
    tokenizer = load_tokenizer(args.model_name_or_path)
    model = load_model(args)
    rope_patcher, scorer, method_config = build_method(args, model, tokenizer, locate_evidence)
    generation = GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=args.do_sample,
                                  temperature=args.temperature, kv_cache_dtype=args.kv_cache_dtype)
    evaluator = evaluator_cls(tokenizer, model, generation, rope_patcher=rope_patcher, scorer=scorer)
    os.makedirs(args.run_dir, exist_ok=True)
    print(f"{args.run_dir}\n{json.dumps({k: v for k, v in method_config.items() if k != 'head_list'})}")
    return evaluator, next(model.parameters()).device, method_config


def _summary(args: argparse.Namespace, method_config: dict, **results) -> dict:
    return {"model": args.model_name_or_path, "method": args.method, "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample, "temperature": args.temperature, **method_config, **results,
            "timestamp": int(time.time())}


def run_nolima(args: argparse.Namespace) -> None:
    tests = load_nolima_tests(args.needle_set_path, args.test_ids)
    evaluator, device, method_config = _setup(args, NoLiMaEvaluator, lambda: NoLiMaNeedleLocator(tests))
    depths = nolima_depths(args.depth_min, args.depth_max, args.depth_intervals)

    per_haystack = []
    for index, haystack in enumerate(tqdm(sorted(os.listdir(args.haystack_dir)), desc="Haystacks")):
        if args.haystacks and haystack not in args.haystacks:
            continue
        placements = NoLiMaPlacements(os.path.join(args.haystack_dir, haystack), index, args.log_placements_dir,
                                      context_length=args.context_length, depths=depths, shift=args.shift,
                                      base_seed=args.base_seed)
        stats = evaluator.evaluate_haystack(placements, tests, args.depth_step, args.run_dir, device)
        per_haystack.append(stats)
        print("  " + " | ".join([stats["haystack"]] + [f"{stats[k]:.2f}% {k}" for k in NOLIMA_METRICS]))

    total = sum(s["num_placements"] for s in per_haystack)
    overall = {k: sum(s[k] * s["num_placements"] for s in per_haystack) / total for k in NOLIMA_METRICS}
    print("Overall | " + " | ".join(f"{overall[k]:.2f}% {k}" for k in NOLIMA_METRICS))
    save_json(os.path.join(args.run_dir, "summary.json"), _summary(
        args, method_config, context_length=args.context_length, depth_step=args.depth_step,
        log_placements_dir=args.log_placements_dir, overall=overall, total_placements=total, per_haystack=per_haystack))


def run_ruler(args: argparse.Namespace) -> None:
    records = {task: load_ruler_records(args.ruler_root, args.tokenizer_tag, task, args.context_length,
                                        args.max_records) for task in args.tasks}
    evaluator, device, method_config = _setup(args, RULEREvaluator, lambda: RulerEvidenceLocator(records))

    per_task = []
    for task in tqdm(args.tasks, desc="Tasks"):
        stats = evaluator.evaluate_task(task, records[task], args.context_length, args.run_dir, device)
        per_task.append(stats)
        print(f"  {task} | {stats['accuracy']:.2f}% ({stats['primary_metric']}) | n={stats['num_records']}")

    total = sum(s["num_records"] for s in per_task)
    overall = sum(s["accuracy"] * s["num_records"] for s in per_task) / total
    print(f"Overall | {overall:.2f}% accuracy")
    save_json(os.path.join(args.run_dir, "summary.json"), _summary(
        args, method_config, context_length=args.context_length, tasks=args.tasks, max_records=args.max_records,
        overall_accuracy=round(overall, 2), per_task=per_task))

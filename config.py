"""Command-line arguments, per-dataset defaults, and run and cache names."""
from __future__ import annotations

import argparse
import os
from typing import Sequence

from QGPM.attn_scorer import HeadSet
from QGPM.region import REGIONS

DATASETS = ("NoLiMa", "RULER")
METHODS = ("Base", "QGPM", "Oracle")

# Default configuration per dataset; every value can be overridden by its flag.
DATASET_DEFAULTS = {
    "NoLiMa": dict(max_new_tokens=1024, segment="sent", lower_topk=3, lower_T=1.0, low_score=0.05, stride_min=0.05,
                   anchor_ratio=1.5),
    "RULER": dict(max_new_tokens=128, segment="chunk", lower_topk=5, lower_T=0.5, low_score=0.1, stride_min=0.4,
                  anchor_ratio=0.375),
}

TASK_ABBR = {
    "qa_1": "qa_1",
    "qa_2": "qa_2",
    "niah_single_1": "niah_sk1",
    "niah_single_2": "niah_sk2",
    "niah_single_3": "niah_sk3",
    "niah_multikey_1": "niah_mk1",
    "niah_multikey_2": "niah_mk2",
    "niah_multikey_3": "niah_mk3",
    "niah_multiquery": "niah_mq",
    "niah_multivalue": "niah_mv",
}


def model_tag(model_name_or_path: str) -> str:
    return model_name_or_path.rstrip("/").split("/")[-1]


def nolima_length_tag(context_length: int) -> str:
    return f"{context_length // 1000}K" if context_length >= 1000 else str(context_length)


def ruler_length_tokens(context_length: str) -> int:
    """Token count of a RULER length tag, e.g. "32k" -> 32000."""
    return int(context_length.lower().rstrip("k")) * 1000


def _add_common_args(p: argparse.ArgumentParser, dataset: str) -> None:
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--results_dir", default="Results/", help="prefix; the run name is appended")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device_map", default="balanced")
    p.add_argument("--no_flash_attn", action="store_true")
    p.add_argument("--kv_cache_dtype", default="none", choices=["none", "int8", "int4", "int2"])
    p.add_argument("--load_in_8bit", action="store_true")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=DATASET_DEFAULTS[dataset]["max_new_tokens"])
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)


def _add_nolima_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--context_length", type=int, required=True, help="tokens per haystack window, e.g. 8000")
    p.add_argument("--needle_set_path", default="Datasets/NoLiMa/data/needlesets/needle_set.json")
    p.add_argument("--haystack_dir", default="Datasets/NoLiMa/data/haystack/rand_shuffle")
    p.add_argument("--log_placements_dir", default="Log_placements/", help="placement cache prefix")
    p.add_argument("--depth_min", type=float, default=0.0)
    p.add_argument("--depth_max", type=float, default=100.0)
    p.add_argument("--depth_intervals", type=int, default=26)
    p.add_argument("--shift", type=int, default=0)
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--depth_step", type=int, default=1, help="every k-th depth of the grid")
    p.add_argument("--test_ids", nargs="+", default=None, help="needle ids, e.g. 0401 0402 (default: all)")
    p.add_argument("--haystacks", nargs="+", default=None, help="haystack files (default: all)")


def _add_ruler_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--context_length", required=True, help="length folder, e.g. 32k")
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--ruler_root", default="Datasets/RULER")
    p.add_argument("--tokenizer_tag", default="qwen", help="data sub-folder under --ruler_root")
    p.add_argument("--max_records", type=int, default=-1, help="first N records per task (default: all)")


def _add_stage2_args(p: argparse.ArgumentParser, dataset: str) -> None:
    defaults = DATASET_DEFAULTS[dataset]
    p.add_argument("--m_star", type=float, default=None, help="gate pair index; overrides --anchor_length")
    p.add_argument("--anchor_length", type=float, default=None,
                   help=f"gate wavelength in tokens (default: {defaults['anchor_ratio']} x context length)")
    p.add_argument("--tau_m", type=float, default=3.0)
    p.add_argument("--stride_min", type=float, default=defaults["stride_min"])
    p.add_argument("--low_score", type=float, default=defaults["low_score"])
    p.add_argument("--region", default="after", choices=REGIONS)


def _add_QGPM_args(p: argparse.ArgumentParser, dataset: str) -> None:
    defaults = DATASET_DEFAULTS[dataset]
    p.add_argument("--heads", default=None, help="head ranking file (default: heads/<model>.txt)")
    p.add_argument("--n_heads", type=int, default=64)
    p.add_argument("--segment", default=defaults["segment"], choices=["sent", "chunk"])
    p.add_argument("--chunk_size", type=int, default=64, help="tokens per sentence-aligned chunk")
    p.add_argument("--lower_topk", type=int, default=defaults["lower_topk"])
    p.add_argument("--lower_T", type=float, default=defaults["lower_T"])
    p.add_argument("--upper_topk", type=int, default=1)
    p.add_argument("--upper_T", type=float, default=0.25)
    p.add_argument("--split_layer", type=int, default=None, help="first layer of the upper map (default: half depth)")
    p.add_argument("--score_cache_dir", default="Score_cache/", help="relevance cache prefix; empty disables it")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse `<dataset> <method> [flags]` and resolve the derived fields."""
    parser = argparse.ArgumentParser(description="Base, QGPM and Oracle runs on NoLiMa and RULER.")
    datasets = parser.add_subparsers(dest="dataset", required=True)
    for dataset in DATASETS:
        methods = datasets.add_parser(dataset).add_subparsers(dest="method", required=True)
        for method in METHODS:
            p = methods.add_parser(method, description=f"{method} on {dataset}")
            _add_common_args(p, dataset)
            (_add_nolima_args if dataset == "NoLiMa" else _add_ruler_args)(p)
            if method != "Base":
                _add_stage2_args(p, dataset)
            if method == "QGPM":
                _add_QGPM_args(p, dataset)
    args = parser.parse_args(argv)
    _resolve(args)
    return args


def _resolve(args: argparse.Namespace) -> None:
    """Set the derived fields `run_dir`, `log_placements_dir`, `anchor_length`, `heads` and `score_cache_path`.

    The run name contains every setting that affects the results; the cache name every input of the cached relevance.
    """
    model = model_tag(args.model_name_or_path)
    if args.dataset == "NoLiMa":
        length = nolima_length_tag(args.context_length)
        needle_set = os.path.splitext(os.path.basename(args.needle_set_path))[0]
        args.log_placements_dir = (f"{args.log_placements_dir}NoLiMa_{length}_{needle_set}_{args.depth_min}_"
                                   f"{args.depth_max}_{args.depth_intervals}_shift{args.shift}_seed{args.base_seed}")
        label = f"NoLiMa_{length}_({model})"
        cache_label = f"NoLiMa_{length}_{needle_set}_shift{args.shift}_seed{args.base_seed}_({model})"
        length_tokens = args.context_length
    else:
        tasks = "+".join(sorted(TASK_ABBR.get(t, t) for t in args.tasks))
        label = f"RULER_{args.context_length}_{tasks}_({model})"
        cache_label = f"RULER_{args.context_length}_({model})"
        length_tokens = ruler_length_tokens(args.context_length)

    name = f"{args.results_dir}{label}_{args.method}"
    if args.method == "QGPM":
        name += _QGPM_tag(args, model, cache_label)
    if args.method != "Base":
        name += _stage2_tag(args, length_tokens)
    if args.dataset == "NoLiMa" and args.depth_step != 1:
        name += f"_depthstep{args.depth_step}"
    args.run_dir = name


def _QGPM_tag(args: argparse.Namespace, model: str, cache_label: str) -> str:
    args.heads = args.heads or f"heads/{model}.txt"
    heads = f"top{args.n_heads}-{HeadSet.load(args.heads, args.n_heads).digest()}"
    segment = "sent" if args.segment == "sent" else f"chunk{args.chunk_size}"
    split = "half" if args.split_layer is None else str(args.split_layer)
    args.score_cache_path = f"{args.score_cache_dir}{cache_label}_{heads}_{segment}.sqlite" if args.score_cache_dir else ""
    return (f"_{heads}_{segment}_lowerK{args.lower_topk}T{args.lower_T:g}_upperK{args.upper_topk}T{args.upper_T:g}"
            f"_split{split}")


def _stage2_tag(args: argparse.Namespace, length_tokens: int) -> str:
    if args.m_star is None and args.anchor_length is None:
        args.anchor_length = DATASET_DEFAULTS[args.dataset]["anchor_ratio"] * length_tokens
    gate = f"mstar{args.m_star:g}" if args.m_star is not None else f"anchor{args.anchor_length:g}"
    return f"_{gate}_taum{args.tau_m:g}_stride{args.stride_min:g}_low{args.low_score:g}_region{args.region}"

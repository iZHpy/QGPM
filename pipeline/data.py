"""NoLiMa needle tests and placements, and RULER records."""
from __future__ import annotations

import json
import os
from functools import cached_property, lru_cache
from typing import Sequence

import numpy as np
from transformers import AutoTokenizer

# All models read the same placements, built with this tokenizer.
PLACEMENT_TOKENIZER = "Qwen/Qwen2.5-7B-Instruct-1M"


def save_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)


def load_nolima_tests(needle_set_path: str, test_ids: Sequence[str] | None = None) -> list[dict]:
    """One test per (needle, question type, test) with the {1}, {2}, ... arguments filled in."""
    with open(needle_set_path) as f:
        needle_set = json.load(f)
    tests = []
    for entry in needle_set:
        for q_type, q_template in entry["questions"].items():
            for test_id, test in entry["tests"].items():
                needle, question = entry["needle"], q_template
                for k, value in enumerate(test["input_args"]):
                    needle = needle.replace("{" + str(k + 1) + "}", value)
                    question = question.replace("{" + str(k + 1) + "}", value)
                tests.append({
                    "test_name": f"{entry['id']}_{test_id}_{q_type}",
                    "needle_id": entry["id"],
                    "task_template": entry["task_template"],
                    "needle": needle,
                    "question": question,
                    "character_set": entry["character_set"],
                    "gold_answers": test.get("gold_answers", ""),
                    "seed_offset": int(entry["id"][:4]),
                })
    if test_ids:
        wanted = set(test_ids)
        tests = [t for t in tests if t["needle_id"] in wanted or t["needle_id"].rstrip("Inv") in wanted]
    return tests


@lru_cache(maxsize=1)
def placement_tokenizer():
    return AutoTokenizer.from_pretrained(PLACEMENT_TOKENIZER)


class BookHaystack:
    """A NoLiMa haystack book in which a needle is placed at a relative depth of a `context_length`-token window."""

    def __init__(self, path: str, tokenizer) -> None:
        with open(path) as f:
            text = f.read()
        self.tokenizer = tokenizer
        self.ids = tokenizer.encode(text, add_special_tokens=False)
        pieces = [self._decode([i]) for i in self.ids]
        self.line_breaks = np.array([0] + [i for i, piece in enumerate(pieces) if "\n" in piece])

    def _decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def place(self, needle: str, context_length: int, depth: float, shift: int) -> tuple[str, int]:
        """Window text with the needle inserted after the first line break at or past the depth, and its token depth."""
        delta = (self.line_breaks - shift) / (context_length + 1) - depth
        delta[delta < 0] = np.inf
        anchor = self.line_breaks[np.argmin(delta)]
        token_depth = int(np.round(context_length * depth))
        start = anchor - token_depth
        pre = self._decode(self.ids[start:anchor + 1])
        pre = pre[:max(pre.rfind("\n"), 0)]
        post = self._decode(self.ids[anchor + 1:start + context_length])
        return pre + " " + needle + "\n" + post, token_depth


class NoLiMaPlacements:
    """Placements of one haystack over the depth grid, cached as one JSON file per (test, depth) in `cache_dir`.

    All placements of a test are built together in depth order, since their {CHAR} draws share one random state.
    """

    def __init__(self, haystack_path: str, haystack_index: int, cache_dir: str, *, context_length: int,
                 depths: Sequence[float], shift: int, base_seed: int) -> None:
        self.haystack_path = haystack_path
        self.name = os.path.splitext(os.path.basename(haystack_path))[0]
        self.haystack_index = haystack_index
        self.cache_dir = os.path.join(cache_dir, self.name)
        self.context_length = context_length
        self.depths = list(depths)
        self.shift = shift
        self.base_seed = base_seed

    @cached_property
    def book(self) -> BookHaystack:
        return BookHaystack(self.haystack_path, placement_tokenizer())

    def _path(self, test: dict, depth: float) -> str:
        return os.path.join(self.cache_dir, f"{test['test_name']}_d{depth:.4f}.json")

    def for_test(self, test: dict) -> list[dict]:
        paths = [self._path(test, depth) for depth in self.depths]
        if all(os.path.exists(p) for p in paths):
            placements = []
            for path, depth in zip(paths, self.depths):
                with open(path) as f:
                    placements.append({**json.load(f), "depth": depth})
            return placements
        os.makedirs(self.cache_dir, exist_ok=True)
        placements = self._build(test)
        for path, placement in zip(paths, placements):
            save_json(path, placement)
        return placements

    def _build(self, test: dict) -> list[dict]:
        rng = np.random.RandomState(self.base_seed + test["seed_offset"] + self.haystack_index)
        placements = []
        for depth in self.depths:
            text, token_depth = self.book.place(test["needle"], self.context_length, depth, self.shift)
            if "{CHAR}" in test["needle"]:
                present = [c for c in test["character_set"] if c in text]
                character = present[0] if present else str(rng.choice(test["character_set"]))
                text = text.replace("{CHAR}", character)
                question = test["question"].replace("{CHAR}", character)
                gold_answers = [character]
            else:
                character, question, gold_answers = None, test["question"], test["gold_answers"] or []
            placements.append({
                "haystack": self.name, "test_name": test["test_name"], "needle": test["needle"], "depth": depth,
                "token_depth": token_depth, "context_length_wo_needle": self.context_length,
                "selected_character": character, "question": question, "gold_answers": gold_answers, "text": text,
            })
        return placements


def nolima_depths(depth_min: float, depth_max: float, depth_intervals: int) -> list[float]:
    """Evenly spaced relative depths in [0, 1]; the bounds are given in percent."""
    return [float(d) / 100.0 for d in np.linspace(depth_min, depth_max, depth_intervals)]


def load_ruler_records(root: str, tokenizer_tag: str, task: str, context_length: str, max_records: int) -> list[dict]:
    """Records of `<root>/<tokenizer_tag>/<task>/<context_length>/validation.jsonl`, the first `max_records` if > 0."""
    with open(os.path.join(root, tokenizer_tag, task, context_length, "validation.jsonl"), encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return records[:max_records] if max_records > 0 else records


def split_ruler_haystack_query(record: dict) -> tuple[str, str]:
    """Haystack and scorer query of a record.

    The query is the text after the haystack, starting at its last "Question:", followed by the answer prefix.
    """
    haystack = record["haystack"]
    tail = record["input"].partition(haystack)[2]
    q_pos = tail.rfind("Question:")
    if q_pos > 0:
        tail = tail[q_pos:]
    return haystack, tail.lstrip("\n") + record["answer_prefix"]

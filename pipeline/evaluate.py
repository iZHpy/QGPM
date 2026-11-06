"""Evaluators: one generation per item, with the item's scores installed when a RoPE patcher is given."""
from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from typing import Any, Optional

import torch
from tqdm import tqdm

from pipeline.data import NoLiMaPlacements, save_json, split_ruler_haystack_query
from pipeline.generate import GenerationConfig, generate_answer
from pipeline.prompts import nolima_prompt, ruler_prompt
from QGPM.scorer import Scorer, ScoringItem

NOLIMA_METRICS = ("EM", "contains", "lastline_EM", "lastline_contains")


def _load_results(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)["results"]


class Evaluator:
    """Base runs without a RoPE patcher (plain RoPE); QGPM and Oracle pass a patcher and a scorer."""

    def __init__(self, tokenizer, model: Any, generation: GenerationConfig, *, rope_patcher: Any = None,
                 scorer: Optional[Scorer] = None) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.generation = generation
        self.rope_patcher = rope_patcher
        self.scorer = scorer

    @contextmanager
    def _scores_installed(self, item: ScoringItem, device: Any):
        """Install the item's scores on the patcher for one generation and clear them afterwards."""
        if self.rope_patcher is None:
            yield
            return
        self.rope_patcher.set_scores(self.scorer.score(item, device))
        try:
            yield
        finally:
            self.rope_patcher.clear_scores()

    def _generate(self, input_ids: torch.Tensor) -> str:
        return generate_answer(self.model, self.tokenizer, input_ids, self.generation)

    def answer(self, item: ScoringItem, device: Any) -> str:
        input_ids = self.tokenizer.encode(item.prompt, return_tensors="pt", add_special_tokens=True).to(device)
        with self._scores_installed(item, device):
            answer = self._generate(input_ids)
        del input_ids
        torch.cuda.empty_cache()
        return answer


def nolima_record(test: dict, placement: dict, raw_answer: str) -> dict:
    gold = placement["gold_answers"]
    last_line = raw_answer.strip().split("\n")[-1]
    return {
        "test_name": test["test_name"],
        "needle": test["needle"],
        "depth": placement["depth"],
        "token_depth": placement["token_depth"],
        "context_length_wo_needle": placement["context_length_wo_needle"],
        "selected_character": placement["selected_character"],
        "gold_answers": gold,
        "question": placement["question"],
        "raw_response": raw_answer,
        "EM": int(raw_answer.strip() in gold),
        "contains": int(any(g in raw_answer for g in gold)),
        "lastline_EM": int(last_line in gold),
        "lastline_contains": int(any(g in last_line for g in gold)),
    }


def nolima_statistics(haystack: str, context_length: int, results: list[dict]) -> dict:
    n = len(results)
    return {
        "haystack": haystack,
        "context_length": context_length,
        "num_placements": n,
        **{key: round(100 * sum(r[key] for r in results) / n, 2) for key in NOLIMA_METRICS},
    }


class NoLiMaEvaluator(Evaluator):
    """Evaluates one haystack into `<results_dir>/<haystack>.json`, resuming by (test_name, depth)."""

    def evaluate_haystack(self, placements: NoLiMaPlacements, tests: list[dict], depth_step: int,
                          results_dir: str, device: Any) -> dict:
        path = os.path.join(results_dir, f"{placements.name}.json")
        results = _load_results(path)
        done = {(r["test_name"], round(r["depth"], 4)) for r in results}
        total = len(tests) * len(placements.depths[::depth_step]) - len(done)
        progress = tqdm(total=total, desc=placements.name, leave=False)
        for test in tests:
            for placement in placements.for_test(test)[::depth_step]:
                q_id = f"{test['test_name']}_{round(placement['depth'], 4)}"
                if (test["test_name"], round(placement["depth"], 4)) in done:
                    continue
                prompt = nolima_prompt(self.tokenizer, test["task_template"], placement["text"], placement["question"])
                item = ScoringItem(placements.name, q_id, placement["question"], placement["text"], prompt)
                results.append(nolima_record(test, placement, self.answer(item, device)))
                save_json(path, {"statistics": {"haystack": placements.name, "context_length": placements.context_length},
                                 "results": results})
                progress.update()
        progress.close()
        statistics = nolima_statistics(placements.name, placements.context_length, results)
        save_json(path, {"statistics": statistics, "results": results})
        return statistics


def ruler_metric(refs: list, raw_answer: str) -> dict:
    """RULER string match of one answer: `part` = any reference found, `all` = fraction of references found."""
    pred = re.sub(r"[\x00-\x1f]", "\n", raw_answer).strip().lower()
    hits = [1.0 if str(r).lower() in pred else 0.0 for r in refs]
    return {"string_match_part": max(hits), "string_match_all": sum(hits) / len(hits)}


def ruler_statistics(task: str, context_length: str, results: list[dict]) -> dict:
    """Aggregate metrics of a task; the accuracy is `string_match_part` on qa tasks and `string_match_all` otherwise."""
    n = len(results)
    primary = "string_match_part" if task.startswith("qa") else "string_match_all"
    totals = {key: sum(r[key] for r in results) for key in ("string_match_part", "string_match_all")}
    return {
        "task": task,
        "context_length": context_length,
        "num_records": n,
        "primary_metric": primary,
        "string_match_part": round(100 * totals["string_match_part"] / n, 2),
        "string_match_all": round(100 * totals["string_match_all"] / n, 2),
        "accuracy": round(100 * totals[primary] / n, 2),
    }


class RULEREvaluator(Evaluator):
    """Evaluates one task into `<results_dir>/<task>.json`, resuming by record position."""

    def evaluate_task(self, task: str, records: list[dict], context_length: str, results_dir: str, device: Any) -> dict:
        path = os.path.join(results_dir, f"{task}.json")
        results = _load_results(path)
        done = {r["record_pos"] for r in results}
        for pos, record in enumerate(tqdm(records, desc=task, leave=False)):
            if pos in done:
                continue
            haystack, query = split_ruler_haystack_query(record)
            item = ScoringItem(task, str(pos), query, haystack, ruler_prompt(self.tokenizer, record))
            raw_answer = self.answer(item, device)
            results.append({"record_pos": pos, "index": record["index"], "outputs": record["outputs"],
                            "raw_response": raw_answer, **ruler_metric(record["outputs"], raw_answer)})
            save_json(path, {"statistics": {"task": task, "context_length": context_length}, "results": results})
        statistics = ruler_statistics(task, context_length, results)
        save_json(path, {"statistics": statistics, "results": results})
        return statistics

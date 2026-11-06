"""Answer generation with HF `generate`."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import QuantizedCache


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int
    do_sample: bool = False
    temperature: float = 1.0
    kv_cache_dtype: str = "none"   # none, int8, int4 or int2


def generate_answer(model, tokenizer, input_ids: torch.Tensor, config: GenerationConfig) -> str:
    kwargs = dict(input_ids=input_ids, max_new_tokens=config.max_new_tokens, do_sample=config.do_sample,
                  pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True)
    if config.do_sample:
        kwargs["temperature"] = config.temperature
    if config.kv_cache_dtype != "none":
        nbits = {"int2": 2, "int4": 4, "int8": 8}[config.kv_cache_dtype]
        kwargs["past_key_values"] = QuantizedCache(backend="hqq" if nbits == 8 else "quanto", config=model.config,
                                                   nbits=nbits, axis_key=0, axis_value=0, q_group_size=64,
                                                   residual_length=128)
    with torch.inference_mode():
        output_ids = model.generate(**kwargs)[0][input_ids.shape[-1]:].tolist()
    return decode_answer(tokenizer, output_ids)


def decode_answer(tokenizer, output_ids: list[int]) -> str:
    """The generated text after the last `</think>`.

    Tokenizers without `</think>` map it to the unknown token, which some of them share with EOS (Granite 3.3 uses
    id 0 for unk, eos and pad); that id is not a `</think>` and must not cut the answer.
    """
    think_end = tokenizer.convert_tokens_to_ids("</think>")
    start = 0
    if think_end is not None and think_end != tokenizer.unk_token_id and think_end in output_ids:
        start = len(output_ids) - output_ids[::-1].index(think_end)
    return tokenizer.decode(output_ids[start:], skip_special_tokens=True).strip()

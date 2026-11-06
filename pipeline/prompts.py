"""Chat prompts for NoLiMa and RULER."""
from __future__ import annotations

NOLIMA_SYSTEM_PROMPT = """You are an AI assistant trained to answer questions based on provided book snippets. 
    You will be given a snippet of text from a book and a question about that text. Your task is to provide 
    a concise and accurate answer to the question using only the information contained in the snippet. 
    Your answer should be based on explicitly stated facts or strong, logical inferences from the snippet. 
    Do not include any information that is not present in the snippet, and do not provide any additional explanation 
    or reasoning in your answer.
"""

RULER_SYSTEM_PROMPT = """You are an AI assistant trained to answer questions based on provided book snippets. 
    You will be given snippets of text and a question about those snippets. Your task is to provide 
    a concise and accurate answer to the question using only the information contained in the snippet. 
    Your answer should be based on explicitly stated facts or strong, logical inferences from the snippet. 
    Do not include any information that is not present in the snippet, and do not provide any additional explanation 
    or reasoning in your answer.
"""


def single_bos(tokenizer, prompt: str) -> str:
    """Drop the template's BOS when encoding adds one anyway (Mistral), so the prompt starts with exactly one."""
    bos = tokenizer.bos_token
    if bos and prompt.startswith(bos) and tokenizer.encode("", add_special_tokens=True)[:1] == [tokenizer.bos_token_id]:
        return prompt[len(bos):]
    return prompt


def chat_prompt(tokenizer, system: str, user: str) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return single_bos(tokenizer, tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))


def nolima_prompt(tokenizer, task_template: str, haystack: str, question: str) -> str:
    return chat_prompt(tokenizer, NOLIMA_SYSTEM_PROMPT, task_template.format(haystack=haystack, question=question))


def ruler_prompt(tokenizer, record: dict) -> str:
    """RULER inputs are already rendered; the answer prefix ends the user turn."""
    return chat_prompt(tokenizer, RULER_SYSTEM_PROMPT, record["input"] + record["answer_prefix"])

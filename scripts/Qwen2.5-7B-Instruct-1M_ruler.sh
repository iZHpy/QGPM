#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

M=Qwen/Qwen2.5-7B-Instruct-1M
for L in 4k 8k 16k 32k 64k; do
    python3 main.py RULER Base --model_name_or_path $M --context_length $L --tasks niah_multikey_3 niah_multivalue qa_1 qa_2 "$@"
done
python3 main.py RULER QGPM --model_name_or_path $M --context_length 4k --tasks niah_multikey_3 qa_1 qa_2 "$@"
python3 main.py RULER QGPM --model_name_or_path $M --context_length 4k --tasks niah_multivalue --lower_T 0 --upper_topk 5 --upper_T 0 "$@"
for L in 8k 16k 64k; do
    python3 main.py RULER QGPM --model_name_or_path $M --context_length $L --tasks niah_multikey_3 qa_1 qa_2 "$@"
    python3 main.py RULER QGPM --model_name_or_path $M --context_length $L --tasks niah_multivalue --upper_topk 4 "$@"
done
python3 main.py RULER QGPM --model_name_or_path $M --context_length 32k --tasks niah_multikey_3 qa_1 qa_2 --m_star 30 "$@"
python3 main.py RULER QGPM --model_name_or_path $M --context_length 32k --tasks niah_multivalue --upper_topk 4 --m_star 30 "$@"

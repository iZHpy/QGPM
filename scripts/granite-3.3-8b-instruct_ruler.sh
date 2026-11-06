#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

M=ibm-granite/granite-3.3-8b-instruct
for L in 4k 8k 16k 32k 64k; do
    python3 main.py RULER Base --model_name_or_path $M --context_length $L --tasks niah_multikey_3 niah_multivalue qa_1 qa_2 "$@"
    python3 main.py RULER QGPM --model_name_or_path $M --context_length $L --tasks niah_multikey_3 --upper_topk 4 "$@"
    python3 main.py RULER QGPM --model_name_or_path $M --context_length $L --tasks niah_multivalue --upper_topk 5 --upper_T 0.5 "$@"
    python3 main.py RULER QGPM --model_name_or_path $M --context_length $L --tasks qa_1 qa_2 --lower_topk 16 --lower_T 8 --upper_topk 16 --upper_T 8 "$@"
done

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

M=Qwen/Qwen2.5-7B-Instruct-1M
for L in 4000 8000 16000 32000; do
    python3 main.py NoLiMa Base --model_name_or_path $M --context_length $L "$@"
done
python3 main.py NoLiMa QGPM --model_name_or_path $M --context_length 4000 "$@"
python3 main.py NoLiMa QGPM --model_name_or_path $M --context_length 8000 --m_star 30 "$@"
python3 main.py NoLiMa QGPM --model_name_or_path $M --context_length 16000 "$@"
python3 main.py NoLiMa QGPM --model_name_or_path $M --context_length 32000 "$@"

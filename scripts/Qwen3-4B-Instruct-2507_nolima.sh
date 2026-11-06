#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

M=Qwen/Qwen3-4B-Instruct-2507
for L in 4000 8000 16000 32000; do
    python3 main.py NoLiMa Base --model_name_or_path $M --context_length $L "$@"
    python3 main.py NoLiMa QGPM --model_name_or_path $M --context_length $L --n_heads 4 "$@"
done

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

M=ibm-granite/granite-3.3-8b-instruct
for L in 4000 8000 16000 32000; do
    python3 main.py NoLiMa Base --model_name_or_path $M --context_length $L "$@"
    python3 main.py NoLiMa QGPM --model_name_or_path $M --context_length $L "$@"
done

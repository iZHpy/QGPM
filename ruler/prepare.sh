#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
RULER_DIR="${RULER_DIR:-$HERE/RULER}"
OUT="$(mkdir -p "${1:-$HERE/../Datasets/RULER/qwen}" && cd "${1:-$HERE/../Datasets/RULER/qwen}" && pwd)"
TOKENIZER="Qwen/Qwen2.5-7B-Instruct"

if [ ! -d "$RULER_DIR" ]; then
    git clone https://github.com/NVIDIA/RULER.git "$RULER_DIR"
    git -C "$RULER_DIR" checkout ab17b78
    git -C "$RULER_DIR" apply "$HERE/ruler.patch"
fi

cd "$RULER_DIR/scripts/data/synthetic/json"
[ -f PaulGrahamEssays.json ] || python3 download_paulgraham_essay.py
{ [ -f squad.json ] && [ -f hotpotqa.json ]; } || bash download_qa_dataset.sh

cd "$RULER_DIR/scripts/data"
for length in 4k 8k 16k 32k 64k; do
    for task in niah_multikey_3 niah_multivalue qa_1 qa_2; do
        python3 prepare.py --save_dir "$OUT" --benchmark synthetic --task "$task" --tokenizer_path "$TOKENIZER" \
            --tokenizer_type hf --max_seq_length $(( ${length%k} * 1024 )) --model_template_type base --num_samples 500
        mkdir -p "$OUT/$task/$length"
        mv "$OUT/$task/validation.jsonl" "$OUT/$task/$length/validation.jsonl"
    done
done

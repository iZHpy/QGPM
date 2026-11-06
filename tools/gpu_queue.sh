#!/usr/bin/env bash
# Sequential GPU job queue for long experiment campaigns (several workers may share one queue).
#
#   tools/gpu_queue.sh worker <gpu> [queue]       run jobs from logs/queue/<queue>/pending in name order
#   AFTER="a b" tools/gpu_queue.sh add <queue-gpu> <name> <cmd...>   enqueue; optional dependencies
#   tools/gpu_queue.sh status [<queue-gpu>]       list pending / running / done / failed jobs
#
# <queue> defaults to gpu<gpu>. A job is one shell command run from the repo root with
# CUDA_VISIBLE_DEVICES=<gpu>, logging to logs/<name>.log. A job whose file starts with
# "#after <names>" waits until those jobs are in done/. Jobs are claimed with an atomic mv, so
# workers can share a queue. POLITE=1: only start a job when the GPU has < 2 GB in use (the GPU
# is shared with other users). `touch logs/queue/<queue>/STOP_gpu<gpu>` ends that worker after
# its current job.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

queue_dir() { echo "logs/queue/$1"; }

gpu_busy() {  # 0 (true) when another process holds >= 2 GB on the GPU
    python3 - "$1" 2>/dev/null <<'PY'
import sys, pynvml
pynvml.nvmlInit()
used = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(int(sys.argv[1]))).used
sys.exit(0 if used >= 2e9 else 1)
PY
}

deps_done() {
    local q=$1 job=$2 dep
    local first; first=$(head -1 "$job")
    [[ "$first" == "#after "* ]] || return 0
    for dep in ${first#\#after }; do [ -f "$q/done/$dep.job" ] || return 1; done
}

worker() {
    local gpu=$1 q; q=$(queue_dir "${2:-gpu$gpu}")
    mkdir -p "$q"/{pending,running,done,failed}
    while [ ! -f "$q/STOP_gpu$gpu" ]; do
        if [ "${POLITE:-0}" = 1 ] && gpu_busy "$gpu"; then sleep 60; continue; fi
        local job="" name="" candidate
        for candidate in $(ls "$q"/pending/*.job 2>/dev/null); do
            deps_done "$q" "$candidate" || continue
            name=$(basename "$candidate" .job)
            if mv "$candidate" "$q/running/" 2>/dev/null; then job=$candidate; break; fi
        done
        if [ -z "$job" ]; then sleep 20; continue; fi
        echo "[$(date -u +%FT%TZ)] start $name gpu$gpu" >> "$q/worker.log"
        CUDA_VISIBLE_DEVICES=$gpu HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 \
            bash "$q/running/$name.job" > "logs/$name.log" 2>&1
        local rc=$?
        local dest=done; [ $rc -ne 0 ] && dest=failed
        mv "$q/running/$name.job" "$q/$dest/"
        echo "[$(date -u +%FT%TZ)] $dest($rc) $name gpu$gpu" >> "$q/worker.log"
    done
}

add() {
    local queue=$1 name=$2; shift 2
    [[ "$queue" =~ ^[0-9]+$ ]] && queue="gpu$queue"
    local q; q=$(queue_dir "$queue"); mkdir -p "$q/pending"
    { [ -n "${AFTER:-}" ] && echo "#after $AFTER"; printf '%s\n' "$*"; } > "$q/pending/$name.job"
    echo "queued $name on $queue${AFTER:+ (after $AFTER)}"
}

status() {
    for q in logs/queue/${1:-*}; do
        [[ "${1:-}" =~ ^[0-9]+$ ]] && q=logs/queue/gpu$1
        echo "== $q"
        for s in running pending failed; do ls "$q/$s" 2>/dev/null | sed "s/^/  $s: /"; done
        echo "  done: $(ls "$q/done" 2>/dev/null | wc -l)"
    done
}

"$@"

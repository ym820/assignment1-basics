#!/usr/bin/env bash
# Phase 2 prep: 100-step runs per batch size to measure tok/s and find the memory max.
# Output: one line per batch size with tok/s, or OOM. Use tok/s to fill TOKENS in bs_sweep.sh.
# Extra args are forwarded to train.py. Env overrides: BS="128 256", PREFIX=calib_owt.

BS=(${BS:-1 8 32 64 256 384 512})
PREFIX=${PREFIX:-calib}

for bs in "${BS[@]}"; do
    if uv run python -m cs336_basics.train --dtype bfloat16 --batch_size "$bs" \
        --max_steps 100 --warmup_iters 10 --log_interval 20 --eval_interval 1000 \
        --no_wandb --prefix "$PREFIX" --name "bs$bs" "$@" > "/tmp/${PREFIX}_bs$bs.log" 2>&1; then
        run=$(ls -d runs/${PREFIX}_bs${bs}_* | tail -n 1)
        tps=$(grep -o '"train/tok_per_sec": [0-9.]*' "$run/metrics.jsonl" | tail -n 1 | grep -o '[0-9.]*$')
        echo "[calib] bs=$bs tok/s=$tps"
    elif grep -q "OutOfMemoryError" "/tmp/${PREFIX}_bs$bs.log"; then
        echo "[calib] bs=$bs OOM"
    else
        echo "[calib] bs=$bs failed, see /tmp/${PREFIX}_bs$bs.log"
    fi
done

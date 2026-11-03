#!/usr/bin/env bash
# Phase 2 prep: 100-step runs per batch size to measure tok/s and find the memory max.
# Output: one line per batch size with tok/s, or OOM. Use tok/s to fill TOKENS in bs_sweep.sh.

BS=(1 8 32 64 256 384 512)

for bs in "${BS[@]}"; do
    if uv run python -m cs336_basics.train --dtype bfloat16 --batch_size "$bs" \
        --max_steps 100 --warmup_iters 10 --log_interval 20 --eval_interval 1000 \
        --no_wandb --prefix calib --name "bs$bs" > "/tmp/calib_bs$bs.log" 2>&1; then
        run=$(ls -d runs/calib_bs${bs}_* | tail -n 1)
        tps=$(grep -o '"train/tok_per_sec": [0-9.]*' "$run/metrics.jsonl" | tail -n 1 | grep -o '[0-9.]*$')
        echo "[calib] bs=$bs tok/s=$tps"
    elif grep -q "OutOfMemoryError" "/tmp/calib_bs$bs.log"; then
        echo "[calib] bs=$bs OOM"
    else
        echo "[calib] bs=$bs failed, see /tmp/calib_bs$bs.log"
    fi
done

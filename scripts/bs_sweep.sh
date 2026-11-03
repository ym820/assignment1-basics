#!/usr/bin/env bash
# Phase 2: batch size sweep, one run at a time. B=128 reuses runs/lr_3e-3_*.
# LR = 3e-3 * sqrt(B / 128). Warmup = 3% of steps; ~20 evals and ~10 checkpoints per run.
# Token budget: full 327.68M where it fits in ~30 min, else tok/s from bs_calib.sh * 1500 s.
# Same resume/skip behavior as lr_sweep.sh.

BS=(1 8 32 64 256 384)  # 384 is the memory max (512 OOMs on A100 40GB)
# Calib tok/s: 1: 6.5K, 8: 52K, 32: 142K, 64: 165K, 256: 194K, 384: 199K
declare -A TOKENS=(
    [1]=9700000 [8]=78700000
    [32]=327680000 [64]=327680000 [256]=327680000 [384]=327680000
)

for bs in "${BS[@]}"; do
    tokens=${TOKENS[$bs]}
    if ! [[ "$tokens" =~ ^[0-9]+$ ]]; then
        echo "[queue] TOKENS[$bs] not set, stopping"; exit 1
    fi

    run=$(ls -d runs/bs_bs${bs}_* 2>/dev/null | tail -n 1)
    if [ -n "$run" ] && { [ -f "$run/checkpoints/final.pt" ] || [ -f "$run/skip" ]; }; then
        echo "[queue] skip bs=$bs ($run)"; continue
    fi

    resume=()
    if [ -n "$run" ]; then
        ckpt=$(ls "$run"/checkpoints/step_*.pt 2>/dev/null | tail -n 1)
        [ -n "$ckpt" ] && resume=(--resume_path "$ckpt")
    fi

    steps=$((tokens / (bs * 256)))
    lr=$(awk -v b="$bs" 'BEGIN { printf "%.2e", 3e-3 * sqrt(b / 128) }')
    warmup=$(( steps * 3 / 100 > 10 ? steps * 3 / 100 : 10 ))
    eval_int=$(( steps / 20 > 1 ? steps / 20 : 1 ))
    ckpt_int=$(( steps / 10 > 1 ? steps / 10 : 1 ))

    echo "[queue] start bs=$bs lr=$lr steps=$steps warmup=$warmup ${resume[*]}"
    uv run python -m cs336_basics.train --dtype bfloat16 --eval_tokens 1000000 \
        --batch_size "$bs" --total_tokens "$tokens" --max_lr "$lr" --warmup_iters "$warmup" \
        --eval_interval "$eval_int" --checkpoint_interval "$ckpt_int" \
        --prefix bs --name "bs$bs" "${resume[@]}"
done

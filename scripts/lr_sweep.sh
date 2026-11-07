#!/usr/bin/env bash
# Phase 1: coarse LR sweep, one run at a time.
# Safe to re-run after a spot preemption: finished runs are skipped, partial runs resume from their latest checkpoint.
# To skip a run you killed on purpose (e.g. diverged), create a marker: touch runs/lr_<lr>_<timestamp>/skip

LRS=(1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1)

for lr in "${LRS[@]}"; do
    run=$(ls -d runs/lr_${lr}_* 2>/dev/null | tail -n 1)
    if [ -n "$run" ] && { [ -f "$run/checkpoints/final.pt" ] || [ -f "$run/skip" ]; }; then
        echo "[queue] skip lr=$lr ($run)"
        continue
    fi

    resume=()
    if [ -n "$run" ]; then
        ckpt=$(ls "$run"/checkpoints/step_*.pt 2>/dev/null | tail -n 1)
        [ -n "$ckpt" ] && resume=(--resume_path "$ckpt")
    fi

    echo "[queue] start lr=$lr ${resume[*]}"
    uv run python -m cs336_basics.train --dtype bfloat16 --eval_tokens 1000000 \
        --max_lr "$lr" --prefix lr --name "$lr" "${resume[@]}"
done

#!/usr/bin/env bash
# Phase 3: architecture ablations (handout 7.3), one run at a time. Baseline reuses runs/lr_3e-3_*.
# Same flags as lr_sweep.sh, so curves are comparable. no_norm also gets lower LRs to test stability.
# Same resume/skip behavior as lr_sweep.sh: touch runs/abl_<name>_<timestamp>/skip to skip a run.

# name:ablation:lr
RUNS=(
    post_norm:post_norm:3e-3
    nope:nope:3e-3
    silu:silu:3e-3
    no_norm_3e-3:no_norm:3e-3
    no_norm_1e-3:no_norm:1e-3
    no_norm_3e-4:no_norm:3e-4
    no_norm_1e-4:no_norm:1e-4
)

for entry in "${RUNS[@]}"; do
    IFS=: read -r name abl lr <<< "$entry"

    run=$(ls -d runs/abl_${name}_* 2>/dev/null | tail -n 1)
    if [ -n "$run" ] && { [ -f "$run/checkpoints/final.pt" ] || [ -f "$run/skip" ]; }; then
        echo "[queue] skip $name ($run)"; continue
    fi

    resume=()
    if [ -n "$run" ]; then
        ckpt=$(ls "$run"/checkpoints/step_*.pt 2>/dev/null | tail -n 1)
        [ -n "$ckpt" ] && resume=(--resume_path "$ckpt")
    fi

    echo "[queue] start $name ablation=$abl lr=$lr ${resume[*]}"
    uv run python -m cs336_basics.train --dtype bfloat16 --eval_tokens 1000000 \
        --max_lr "$lr" --ablation "$abl" --prefix abl --name "$name" "${resume[@]}"
done

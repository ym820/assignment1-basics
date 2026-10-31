# Experiment plan

## Compute budget (from the handout, section 7)

A baseline TinyStories run should take 20-30 min on 1 B200.

| Problem | B200 hours |
| --- | --- |
| `learning_rate` | 2 |
| `batch_size_experiment` | 1 |
| `layer_norm_ablation` | 0.5 |
| `pre_norm_ablation` | 0.5 |
| `no_pos_emb` | 0.5 |
| `swiglu_ablation` | 0.5 |
| `main_experiment` (OWT) | 2 |
| **Core total** | **7** |
| `leaderboard` (optional, 45 min per run on B200) | 10 |

Target: TinyStories val loss <= 1.45 (low-resource target: 2.00 at 40M tokens).

## Hardware: GCP A100 40GB (spot)

- Paid from GCP credits. Free-trial accounts cannot use GPUs - upgrade to a paid billing account (credits carry over) and request GPU quota under IAM & Admin > Quotas early.
- Price (Oct 2026): ~$2.20/hr spot, ~$3.67/hr on-demand. Check the official GCP pricing page for your region.
- Estimated speed: 3-5x slower than B200 (spec ratio ~7x compute, ~5x bandwidth; small model won't saturate either GPU). Unmeasured.

| Experiment | B200 (handout) | A100 estimate |
| --- | --- | --- |
| One baseline run | 20-30 min | ~1-2.5 h |
| LR sweep | 2 h | ~6-10 h |
| Batch size | 1 h | ~3-5 h |
| 4 ablations | 2 h | ~6-10 h |
| OWT main | 2 h | ~6-10 h |
| **Core total** | **7 h** | **~20-35 h** |

Calibrate: time 100 steps on the A100, multiply by 100 for a full 10K-step run, and rescale this table.

## Base config (TinyStories)

- Handout: `vocab_size=10000`, `context_length=256`, `d_model=512`, `d_ff=1344`, `rope_theta=10000`, `num_layers=4`, `num_heads=16`, total tokens 327,680,000
- `batch_size=128`, `max_steps = 327,680,000 / (128 * 256) = 10,000`
- `cosine_cycle_iters = max_steps`, `warmup_iters` ~300-500
- Fallback: `batch_size=64`, `max_steps=20,000` (same token budget)

## Memory (A100 40GB)

From the `adamw_accounting` formula:

- TinyStories: ~0.37 GB fixed + ~94 MB per sequence. B=128 needs ~12 GB. Realistic max B ~256.
- OWT (V=32K): ~0.73 GB fixed + ~139 MB per sequence. B=128 needs ~18 GB. Realistic max B ~256.

## Schedule (48-72 h wall clock)

1. **Day 0 (local, Mac MPS):** debug until loss drops and checkpoint resume works. No GPU cost.
2. **Day 1:** LR sweep. Everything else reuses the best LR. Kill divergent runs early.
3. **Day 2:** batch-size sweep (1 up to memory limit, including 64 and 128) + 4 ablations, in parallel.
4. **Day 3:** OWT main experiment + reruns.

## Run design

Naming: `--prefix` per phase (`lr`, `bs`, `abl`, `owt`), `--name` per run (`3e-3`, `bs64`, `nope`).

### Phase 0: GPU calibration (~30 min)

- 200-step smoke test with `--dtype bfloat16` and W&B on. Check loss ~9.2 at step 0, W&B panels, memory in `nvidia-smi`.
- Concurrency test: run 1, 2, 4 copies at once and sum `tok_per_sec`. Where the sum stops growing sets the jobs-per-GPU for later phases.

### Phase 1: LR sweep

- Coarse pass: log-spaced LRs around `1e-3`. Open: range and number of points.
- Fine pass: 2-3 points around the coarse winner.
- Include at least one divergent LR (part b, edge of stability). Kill divergent runs early.
- Run sweep points concurrently (same load, so comparable to each other).
- Open: full 10K steps for every point, or shorter runs for the coarse pass? The best LR can shift with run length.

### Phase 2: batch size

- Sizes: 1, a few small ones, 64, 128, 256, memory max.
- At a fixed 327.68M tokens, B=1 needs 1.28M steps, so small batches need a different stopping rule. Open: fixed steps, fixed wall-clock, or fixed tokens with truncated runs.
- Re-tune LR per batch size (1-2 points each, not a full sweep).
- Scale `warmup_iters` with `max_steps`.
- Throughput is part of the result: run alone or all at the same concurrency.

### Phase 3: ablations

- Needs model switches not yet in `train.py` / `model.py`: no RMSNorm, post-norm, NoPE, SiLU FFN with `d_ff = 4 * d_model`. Build and smoke-test locally first.
- Run all four at the best LR concurrently, plus a baseline rerun in the same batch for fair wall-clock comparison.
- No-RMSNorm: 1-2 extra runs at lower LR.

### Phase 4: OWT

- `--vocab_size 32000`, OWT data paths, same step count.
- 1-2 LR checks, since the best TinyStories LR may not transfer.

Total: ~20-30 runs. Concurrency from phase 0 decides whether this fits the 48-72 h window.

## Running unattended

- Queue each phase's runs in one script, launched in `tmux`.
- Run 2-3 jobs at once on one A100 (each ~12 GB at B=128), or use 2 A100 VMs to halve wall clock at the same cost.
- Spot VMs can be preempted: checkpoint often and make the queue script auto-resume from the latest checkpoint.
- Log to W&B with steps and wall-clock time.

import argparse
import json
import os
import time
import math
import numpy as np
import torch
from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.data import get_batch, get_eval_batch
from cs336_basics.nn_utils import cross_entropy, gradient_clipping
from cs336_basics.training import load_checkpoint, save_checkpoint

def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--train_data_path", type=str, default="outputs/training/tinystories_train.uint16.bin")
    parser.add_argument("--val_data_path", type=str, default="outputs/training/tinystories_valid.uint16.bin")
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)

    # Model
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")

    # Optimizer
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=None)  # derived: 10% of max_lr
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.1)

    # Schedule
    parser.add_argument("--warmup_iters", type=int, default=300)
    parser.add_argument("--cosine_cycle_iters", type=int, default=None)  # derived: max_steps

    # Training
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total_tokens", type=int, default=327680000)
    parser.add_argument("--max_steps", type=int, default=None)  # derived: total_tokens // (batch_size * context_length)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_tokens", type=int, default=None)  # cap for periodic evals (evenly spaced windows); None = full pass. Final eval is always full
    parser.add_argument("--eval_batch_size", type=int, default=128)  # independent of batch_size, so small-batch runs still eval fast
    parser.add_argument("--checkpoint_interval", type=int, default=1000)
    parser.add_argument("--resume_path", type=str, default=None)

    # Run
    parser.add_argument("--run_dir", type=str, default="runs")
    parser.add_argument("--prefix", type=str, default="ts_baseline")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="cs336-a1")

    args = parser.parse_args()
    return args

def resolve_args(args):
    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    print(f"Using device: {args.device}")

    assert args.d_model % args.num_heads == 0, f"d_model ({args.d_model}) must be divisible by num_heads ({args.num_heads})"

    if args.max_steps is None:
        # Calculate max_steps based on total token budget, batch size, and context length
        args.max_steps = args.total_tokens // (args.batch_size * args.context_length)

    if args.cosine_cycle_iters is None:
        # Set cosine cycle iters to the max steps
        args.cosine_cycle_iters = args.max_steps
    assert args.warmup_iters < args.cosine_cycle_iters, f"warmup_iters ({args.warmup_iters}) must be less than cosine_cycle_iters ({args.cosine_cycle_iters})"
    
    if args.min_lr is None:
        # Set min_lr to 10% of max_lr
        args.min_lr = 0.1 * args.max_lr
    
    if args.d_ff is None:
        args.d_ff = 64 * math.floor((args.d_model * 8 / 3) / 64)

    return args

def evaluate(model: torch.nn.Module, data: np.memmap, device, args, full: bool = False):
    """Mean per-token loss over non-overlapping windows: all of them if full or no cap, else an evenly spaced subset of ~eval_tokens."""
    model.eval()
    total_windows = (data.shape[0] - 1) // args.context_length
    assert total_windows > 0, f"Dataset too short for context_length ({args.context_length})"
    if full or args.eval_tokens is None:
        window_stride = 1
    else:
        max_windows = max(1, args.eval_tokens // args.context_length)
        window_stride = math.ceil(total_windows / max_windows)
    num_windows = math.ceil(total_windows / window_stride)
    num_batches = math.ceil(num_windows / args.eval_batch_size)

    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.dtype=="bfloat16"):
        for batch_idx in range(num_batches):
            inputs, targets = get_eval_batch(
                dataset=data,
                batch_idx=batch_idx,
                batch_size=args.eval_batch_size,
                context_length=args.context_length,
                device=device,
                window_stride=window_stride
            )
            loss = cross_entropy(model(inputs), targets)
            # Weight by token count so the smaller last batch isn't over-counted
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()
    model.train()
    return total_loss / total_tokens, total_tokens

def log(metrics_path, _wandb, metrics):
    with open(metrics_path, "a") as f:
        f.write(json.dumps(metrics) + "\n")
    
    if _wandb is not None:
        _wandb.log(metrics, step=metrics["step"])

def main():
    args = parse_args()
    args = resolve_args(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.resume_path:
        # runs/<run_name>/checkpoints/step_XXXXXX.pt -> runs/<run_name>
        run_path = os.path.dirname(os.path.dirname(os.path.abspath(args.resume_path)))
        run_name = os.path.basename(run_path)
    else:
        suffix = f"{args.name}_" if args.name else ""
        run_name = f"{args.prefix}_{suffix}{time.strftime('%Y%m%d_%H%M%S')}"
        run_path = os.path.join(args.run_dir, run_name)

    ckpt_dir = os.path.join(run_path, "checkpoints/")
    metrics_path = os.path.join(run_path, "metrics.jsonl")
    os.makedirs(ckpt_dir, exist_ok=True)

    config_path = os.path.join(run_path, "config.json")
    if not args.resume_path:
        with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=2)

    _wandb = None
    if not args.no_wandb:
        import wandb
        _wandb = wandb
        _wandb.init(
            project=args.wandb_project,
            name=run_name,
            id=run_name,
            resume="allow",
            config=vars(args),
            dir=run_path
        )
    
    # Load memory-mapped dataset
    # uint16 holds 0 to 65535, which covers this experiments' vocab (10K TinyStories, 32K OWT). It takes 2 types per token
    # int32 holds about 2.1B, so it's used for training files with more vocabs but it costs 4 bytes per token
    train_mm = np.memmap(args.train_data_path, dtype=np.uint16, mode="r")
    val_mm = np.memmap(args.val_data_path, dtype=np.uint16, mode="r")
    device = torch.device(args.device)

    # Create model
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
        dtype=torch.float32
    )
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Create optimizer
    optimizer = AdamW(
        params=model.parameters(),
        lr=args.max_lr,
        betas=args.betas,
        weight_decay=args.weight_decay,
        eps=args.eps
    )

    # Load checkpoint from resume_path
    start_iteration = 0
    if args.resume_path:
        start_iteration = load_checkpoint(args.resume_path, model, optimizer) + 1
        np.random.seed(args.seed + start_iteration)
        torch.manual_seed(args.seed + start_iteration)
        print(f"[resume] loaded checkpoint from {args.resume_path}, starting iteration at {start_iteration}")

    # Continue wall-clock time from the last entry logged at or before the checkpoint
    time_offset = 0.0
    if args.resume_path and os.path.exists(metrics_path):
        with open(metrics_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:  # line cut off by preemption
                    continue
                if entry["step"] < start_iteration:
                    time_offset = max(time_offset, entry["time_s"])

    # Training loop
    start_time = time.perf_counter()
    eval_time_total = 0.0
    nan_alerted = False
    for step in range(start_iteration, args.max_steps):

        # Set optimizer's lr with scheduler
        lr = get_lr_cosine_schedule(
            it=step,
            max_learning_rate=args.max_lr,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.cosine_cycle_iters
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        
        # Get a batch
        inputs, targets = get_batch(
            dataset=train_mm,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=device
        )

        # Foward pass and loss
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.dtype=="bfloat16"):
            logits = model(inputs) # (B, S, V)
            loss = cross_entropy(logits=logits, targets=targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        grad_norm = gradient_clipping(model.parameters(), max_l2_norm=args.max_grad_norm)

        # Optimizer step
        optimizer.step()

        # Periodic hooks
        if step % args.log_interval == 0:
            train_loss = loss.item()
            train_grad_norm = grad_norm.item()
            session_time = time.perf_counter() - start_time
            elapsed = time_offset + session_time
            train_time = session_time - eval_time_total
            tokens_per_it = args.batch_size * args.context_length
            tokens_seen = (step + 1) * tokens_per_it
            tok_per_sec = (step + 1 - start_iteration) * tokens_per_it / train_time
            metrics = {"step": step, "tokens": tokens_seen, "time_s": elapsed, "train/loss": train_loss, "train/lr": lr, "train/grad_norm": train_grad_norm, "train/tok_per_sec": tok_per_sec, "train/time_s": train_time}
            print(
                f"[train] step {step}/{args.max_steps} | loss {train_loss:.4f} | lr {lr:.2e} | "
                f"grad_norm {train_grad_norm:.3f} | tokens {tokens_seen / 1e6:.1f}M | "
                f"{tok_per_sec / 1e3:.1f}K tok/s | elapsed {elapsed / 60:.1f} min"
            )
            log(metrics_path, _wandb, metrics)

            # Alert once if training diverged
            if _wandb is not None and not nan_alerted and not math.isfinite(train_loss):
                _wandb.alert(title=f"{run_name} diverged", text=f"train loss {train_loss} at step {step}", level=_wandb.AlertLevel.WARN)
                nan_alerted = True

        if step % args.eval_interval == 0 or step == args.max_steps-1:
            eval_start = time.perf_counter()
            # Wall-clock time up to this eval (includes earlier evals and pre-resume time)
            elapsed = time_offset + (eval_start - start_time)

            # Evaluate and its duration
            is_final = step == args.max_steps - 1
            val_loss, val_tokens = evaluate(model, val_mm, device, args, full=is_final)
            val_ppl = math.exp(val_loss)
            eval_time_s = time.perf_counter() - eval_start
            eval_time_total += eval_time_s

            tokens_seen = (step + 1) * args.batch_size * args.context_length
            metrics = {"step": step, "tokens": tokens_seen, "time_s": elapsed, "val/loss": val_loss, "val/ppl": val_ppl, "val/time_s": eval_time_s, "val/tokens": val_tokens, "val/full": is_final}
            if is_final:
                print(f"[Final eval]  step {step}/{args.max_steps} | val_loss {val_loss:.4f} | val_ppl {val_ppl:.4f} | eval_time_s {eval_time_s:.1f} s")
                if _wandb is not None:
                    # Handout target applies to TinyStories only
                    target = f" (target <= 1.45: {'yes' if val_loss <= 1.45 else 'no'})" if "tinystories" in args.val_data_path else ""
                    _wandb.alert(
                        title=f"{run_name} finished",
                        text=f"val_loss {val_loss:.4f}{target} | ppl {val_ppl:.2f} | {step + 1} steps, {tokens_seen / 1e6:.1f}M tokens | {elapsed / 60:.1f} min",
                        level=_wandb.AlertLevel.INFO
                    )
            else:
                print(f"[eval]  step {step}/{args.max_steps} | val_loss {val_loss:.4f} | val_ppl {val_ppl:.4f} | eval_time_s {eval_time_s:.1f} s")
            log(metrics_path, _wandb, metrics)

        if step > 0 and step % args.checkpoint_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"step_{step:06d}.pt")
            save_checkpoint(model, optimizer, step, ckpt_path)
            print(f"[ckpt]  step {step} | saved to {ckpt_path}")
    
    # Final checkpoint (skipped if resumed from a finished run, so the loop never ran)
    if start_iteration < args.max_steps:
        ckpt_path = os.path.join(ckpt_dir, "final.pt")
        save_checkpoint(model, optimizer, args.max_steps - 1, ckpt_path)
    if _wandb is not None:
        _wandb.finish()

if __name__ == "__main__":
    main()
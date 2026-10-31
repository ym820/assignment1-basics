import argparse
import json
import os
import time
import math
import numpy as np
import torch
from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.data import get_batch
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
    parser.add_argument("--eval_iters", type=int, default=20)
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

def evaluate(model: torch.nn.Module, data: np.memmap, device, args):
    model.eval()
    losses = []
    for _ in range(args.eval_iters):
        val_inputs, val_targets = get_batch(
            dataset=data,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=device
        )
        with torch.no_grad():
            logits = model(val_inputs)
            loss = cross_entropy(logits, val_targets)
        
        losses.append(loss.item())
    model.train()
    return np.mean(losses)

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

    if args.name is not None:
        run_name = f"{args.prefix}_{args.name}_{time.strftime("%Y%m%d_%H%M%S")}"
    else:
        run_name = f"{args.prefix}_{time.strftime("%Y%m%d_%H%M%S")}"
    
    run_path = os.path.join(args.run_dir, run_name)
    ckpt_dir = os.path.join(run_path, "checkpoints/")
    metrics_path = os.path.join(run_path, "metrics.jsonl")
    os.makedirs(ckpt_dir, exist_ok=True)

    config_path = os.path.join(run_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    _wandb = None
    if not args.no_wandb:
        import wandb
        _wandb = wandb
        _wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            dir=run_path
        )
    
    # Load memory-mapped dataset
    # uint16 holds 0 to 65535, which covers this experiments' vocab (10K TinyStories, 32K OWT). It takes 2 types per token
    # int32 holds about 2.1B, so it's used for training files with more vocabs but it costs 4 bytes per token
    train_mm = np.memmap(args.train_data_path, dtype=np.uint16, mode="r")
    val_mm = np.memmap(args.val_data_path, dtype=np.uint16, mode="r")
    
    # Create model
    device = torch.device(args.device)

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
        print(f"[resume] loaded checkpoint from {args.resume_path}, starting iteration at {start_iteration}")
    
    # Training loop
    start_time = time.perf_counter()
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
            elapsed = time.perf_counter() - start_time
            tokens_per_it = args.batch_size * args.context_length
            tokens_seen = (step + 1) * tokens_per_it
            tok_per_sec = (step + 1 - start_iteration) * tokens_per_it / elapsed
            metrics = {"step": step, "tokens": tokens_seen, "time_s": elapsed, "train/loss": train_loss, "train/lr": lr, "train/grad_norm": train_grad_norm, "train/tok_per_sec": tok_per_sec}
            print(
                f"[train] step {step}/{args.max_steps} | loss {train_loss:.4f} | lr {lr:.2e} | "
                f"grad_norm {train_grad_norm:.3f} | tokens {tokens_seen / 1e6:.1f}M | "
                f"{tok_per_sec / 1e3:.1f}K tok/s | elapsed {elapsed / 60:.1f} min"
            )
            log(metrics_path, _wandb, metrics)

        if step % args.eval_interval == 0 or step == args.max_steps-1:
            eval_start = time.perf_counter()
            # Wall-clock time (without evaluate)
            elapsed = eval_start - start_time

            # Evaluate and its duration
            val_loss = evaluate(model, val_mm, device, args)
            val_ppl = math.exp(val_loss)
            eval_time_s = time.perf_counter() - eval_start

            tokens_seen = (step + 1) * args.batch_size * args.context_length
            metrics = {"step": step, "tokens": tokens_seen, "time_s": elapsed, "val/loss": val_loss, "val/ppl": val_ppl, "val/time_s": eval_time_s}
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
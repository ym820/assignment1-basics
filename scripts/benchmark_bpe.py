"""Fair head-to-head benchmark of two BPE training implementations.

This is *tooling*, not part of the assignment: it treats `train_bpe` as a black
box and only measures it. It never touches BPE logic.

Why a separate script instead of extending run_train_bpe.py:
  - Each timed run happens in a *fresh subprocess* so there is no warm import
    state or leftover memory from a previous run, and so peak memory can be
    measured on a clean process tree.
  - tracemalloc (used by run_train_bpe.py) only sees the main Python process.
    Both variants spawn multiprocessing workers, whose memory it cannot see.
    Here we poll RSS of the whole process tree with psutil instead.

Methodology knobs it controls:
  - warms the OS file cache once so every run sees warm I/O equally,
  - runs each variant N times and reports the median (one run is noise),
  - alternates run order (ABBA...) so thermal drift does not favor one variant,
  - times only the train_bpe call with perf_counter, with no profiler attached,
  - verifies both variants produce identical vocab+merges before timing, since
    comparing the speed of two things that compute differently is meaningless.

Usage:
    uv run scripts/benchmark_bpe.py --dataset tinystories
    uv run scripts/benchmark_bpe.py --input data/foo.txt --vocab-size 10000
    uv run scripts/benchmark_bpe.py --dataset owt --repeats 3 \
        --equality-input data/TinyStoriesV2-GPT4-valid.txt
"""

import argparse
import importlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

import psutil

DATASETS = {
    "tinystories": {
        "input": "data/TinyStoriesV2-GPT4-train.txt",
        "vocab_size": 10_000,
        "special_tokens": ["<|endoftext|>"],
    },
    "owt": {
        "input": "data/owt_train.txt",
        "vocab_size": 32_000,
        "special_tokens": ["<|endoftext|>"],
    },
}

VARIANTS = ["train_bpe_parallel", "train_bpe"]
SAMPLE_INTERVAL = 0.05  # seconds between RSS samples


def load_train_bpe(variant):
    """Import cs336_basics.<variant>.train_bpe."""
    module = importlib.import_module(f"cs336_basics.{variant}")
    return module.train_bpe


# --------------------------------------------------------------------------- #
# Child mode: run one training, measure just the train_bpe call, write result. #
# --------------------------------------------------------------------------- #
def run_child(args):
    train_bpe = load_train_bpe(args.variant)
    special_tokens = json.loads(args.special_tokens)
    t0 = time.perf_counter()
    train_bpe(
        input_path=args.input,
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
    )
    elapsed = time.perf_counter() - t0
    with open(args.out, "w") as f:
        json.dump({"seconds": elapsed}, f)


# --------------------------------------------------------------------------- #
# Parent mode helpers.                                                         #
# --------------------------------------------------------------------------- #
def warm_cache(path):
    """Read the file once so subsequent runs hit warm OS cache."""
    with open(path, "rb") as f:
        while f.read(1 << 20):
            pass


def tree_rss(proc):
    """Sum RSS over a process and all its descendants (bytes).

    Over-counts shared pages (each process's RSS includes shared libraries), so
    this is an upper bound, not an exact unique footprint. It is still far more
    honest than tracemalloc, which sees none of the worker processes.
    """
    total = 0
    try:
        total += proc.memory_info().rss
    except psutil.NoSuchProcess:
        return total
    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total


def timed_run(variant, args):
    """Launch one isolated training subprocess; return (seconds, peak_rss_bytes)."""
    fd, out_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    cmd = [
        sys.executable, __file__,
        "--_child",
        "--variant", variant,
        "--input", args.input,
        "--vocab-size", str(args.vocab_size),
        "--special-tokens", json.dumps(args.special_tokens),
        "--_out", out_path,
    ]
    # Inherit stdout/stderr so the variant's own progress output still shows,
    # and so a full PIPE buffer can never deadlock a long run.
    proc = subprocess.Popen(cmd)
    ps = psutil.Process(proc.pid)
    peak = 0
    while proc.poll() is None:
        peak = max(peak, tree_rss(ps))
        time.sleep(SAMPLE_INTERVAL)
    if proc.returncode != 0:
        os.unlink(out_path)
        sys.exit(f"variant {variant!r} exited with code {proc.returncode}")
    with open(out_path) as f:
        seconds = json.load(f)["seconds"]
    os.unlink(out_path)
    return seconds, peak


def check_equality(args):
    """Run both variants once in-process and assert identical output."""
    print(f"Equality check on {args.equality_input} (vocab {args.vocab_size}) ...")
    results = {}
    for variant in VARIANTS:
        train_bpe = load_train_bpe(variant)
        vocab, merges = train_bpe(
            input_path=args.equality_input,
            vocab_size=args.vocab_size,
            special_tokens=args.special_tokens,
        )
        results[variant] = (vocab, merges)
    (va, ma), (vb, mb) = results[VARIANTS[0]], results[VARIANTS[1]]
    if va != vb:
        sys.exit(f"MISMATCH: vocabs differ ({len(va)} vs {len(vb)} entries)")
    if ma != mb:
        n = min(len(ma), len(mb))
        first = next((i for i in range(n) if ma[i] != mb[i]), n)
        sys.exit(
            f"MISMATCH: merges differ (len {len(ma)} vs {len(mb)}, "
            f"first diff at index {first})"
        )
    print(f"  OK: identical vocab ({len(va)}) and merges ({len(ma)}).\n")


def schedule(repeats):
    """Alternate order per repeat: AB, BA, AB, ... to spread thermal drift."""
    a, b = VARIANTS
    order = []
    for i in range(repeats):
        order.extend([a, b] if i % 2 == 0 else [b, a])
    return order


# --------------------------------------------------------------------------- #
# Args + entry.                                                                #
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=DATASETS.keys())
    p.add_argument("--input")
    p.add_argument("--vocab-size", type=int)
    p.add_argument("--repeats", type=int, default=3, help="Runs per variant (default 3).")
    p.add_argument("--equality-input",
                   help="File for the equality check (default: --input). "
                        "Point at a small file (e.g. the valid set) to keep it cheap.")
    p.add_argument("--skip-equality", action="store_true")

    # Hidden child-mode plumbing.
    p.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--variant", choices=VARIANTS, help=argparse.SUPPRESS)
    p.add_argument("--special-tokens", help=argparse.SUPPRESS)
    p.add_argument("--_out", dest="out", help=argparse.SUPPRESS)

    args = p.parse_args()

    if args._child:
        return args

    cfg = DATASETS.get(args.dataset, {})
    args.input = args.input or cfg.get("input")
    args.vocab_size = args.vocab_size or cfg.get("vocab_size")
    args.special_tokens = cfg.get("special_tokens", ["<|endoftext|>"])
    args.equality_input = args.equality_input or args.input
    if not args.input or not args.vocab_size:
        p.error("need --dataset, or both --input and --vocab-size")
    return args


def report(times, peaks):
    print("\n" + "=" * 64)
    print(f"{'variant':<22}{'median s':>10}{'min s':>10}{'peak RSS MB':>14}")
    print("-" * 64)
    for v in VARIANTS:
        t = times[v]
        pk = max(peaks[v]) / 1024 / 1024
        print(f"{v:<22}{statistics.median(t):>10.1f}{min(t):>10.1f}{pk:>14.0f}")
    print("=" * 64)
    fast, slow = sorted(VARIANTS, key=lambda v: statistics.median(times[v]))
    speedup = statistics.median(times[slow]) / statistics.median(times[fast])
    print(f"{fast} is {speedup:.2f}x faster (median). "
          f"Raw times: {json.dumps({v: [round(x, 1) for x in times[v]] for v in VARIANTS})}")
    print("Peak RSS sums per-process RSS across the tree, so it over-counts "
          "shared pages: read it as an upper bound.")


def main():
    args = parse_args()
    if args._child:
        run_child(args)
        return

    print(f"Input: {args.input} ({os.path.getsize(args.input) / 1024 / 1024:.1f} MB)")
    print(f"Vocab {args.vocab_size}, {args.repeats} repeats/variant, order-alternated.\n")

    if not args.skip_equality:
        check_equality(args)

    print("Warming OS file cache ...")
    warm_cache(args.input)

    times = {v: [] for v in VARIANTS}
    peaks = {v: [] for v in VARIANTS}
    for i, variant in enumerate(schedule(args.repeats), 1):
        secs, peak = timed_run(variant, args)
        times[variant].append(secs)
        peaks[variant].append(peak)
        print(f"[{i}/{args.repeats * len(VARIANTS)}] {variant:<22} "
              f"{secs:>7.1f}s  peak {peak / 1024 / 1024:>6.0f} MB")

    report(times, peaks)


if __name__ == "__main__":
    main()

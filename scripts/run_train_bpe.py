import argparse
import cProfile
import json
import os
import pstats
import time
import tracemalloc

from cs336_basics.train_bpe_parallel import train_bpe

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


def parse_args():
    p = argparse.ArgumentParser(description="Train BPE on a dataset.")
    p.add_argument("--dataset", choices=DATASETS.keys(), help="Predefined dataset.")
    p.add_argument("--input", help="Input file (overrides --dataset).")
    p.add_argument("--vocab-size", type=int, help="Vocab size (overrides --dataset default).")
    p.add_argument("--output-dir", default="outputs/bpe")
    p.add_argument("--profile", action="store_true", help="Run under cProfile, save .prof, print top-N.")
    p.add_argument("--top-n", type=int, default=20)
    args = p.parse_args()

    cfg = DATASETS.get(args.dataset, {})
    args.input = args.input or cfg.get("input")
    args.vocab_size = args.vocab_size or cfg.get("vocab_size")
    args.special_tokens = cfg.get("special_tokens", ["<|endoftext|>"])
    args.name = args.dataset or os.path.splitext(os.path.basename(args.input or ""))[0]

    if not args.input or not args.vocab_size:
        p.error("need --dataset, or both --input and --vocab-size")

    return args


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Input: {args.input} ({os.path.getsize(args.input) / 1024 / 1024:.1f} MB)")
    print(f"Vocab size target: {args.vocab_size}")

    profiler = cProfile.Profile() if args.profile else None
    tracemalloc.start()
    t0 = time.time()

    if profiler:
        profiler.enable()
    vocab, merges = train_bpe(
        input_path=args.input,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
    )
    if profiler:
        profiler.disable()

    elapsed = time.time() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Peak memory: {peak / 1024 / 1024:.1f} MB")
    print(f"Vocab size: {len(vocab)}, Merges: {len(merges)}")
    print(f"Longest token: {max(vocab.values(), key=len)}")

    suffix = f"{args.name}_{args.vocab_size}"
    vocab_path = os.path.join(args.output_dir, f"bpe_vocab_{suffix}.json")
    merges_path = os.path.join(args.output_dir, f"bpe_merges_{suffix}.txt")

    with open(vocab_path, "w") as f:
        json.dump({str(k): v.decode("latin-1") for k, v in vocab.items()}, f, indent=2)
    with open(merges_path, "w") as f:
        for t1, t2 in merges:
            f.write(t1.decode("latin-1") + " " + t2.decode("latin-1") + "\n")

    print(f"Saved vocab → {vocab_path}")
    print(f"Saved merges → {merges_path}")

    if profiler:
        prof_path = os.path.join(args.output_dir, f"profile_{suffix}.prof")
        profiler.dump_stats(prof_path)
        stats = pstats.Stats(profiler).sort_stats("cumulative")
        print(f"\nTop {args.top_n} functions by cumulative time:")
        stats.print_stats(args.top_n)
        print(f"Saved profile → {prof_path}")


if __name__ == "__main__":
    main()

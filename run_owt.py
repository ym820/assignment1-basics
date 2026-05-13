import os
import json
import time
import tracemalloc
from cs336_basics.train_bpe import train_bpe

VOCAB_SIZE = 32_000
SPECIAL_TOKENS = ["<|endoftext|>"]
INPUT_PATH = "data/owt_train.txt"
OUTPUT_DIR = "outputs/bpe"
os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
    print(f"Input: {INPUT_PATH} ({os.path.getsize(INPUT_PATH) / 1024 / 1024:.1f} MB)")

    tracemalloc.start()
    t0 = time.time()

    vocab, merges = train_bpe(
        input_path=INPUT_PATH,
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
    )

    elapsed = time.time() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Peak memory: {peak / 1024 / 1024:.1f} MB")
    print(f"Vocab size: {len(vocab)}, Merges: {len(merges)}")
    print(f"Longest token: {max(vocab.values(), key=len)}")

    vocab_path = os.path.join(OUTPUT_DIR, "owt_vocab.json")
    merges_path = os.path.join(OUTPUT_DIR, "owt_merges.txt")

    with open(vocab_path, "w") as f:
        json.dump({str(k): v.decode("latin-1") for k, v in vocab.items()}, f, indent=2)

    with open(merges_path, "w") as f:
        for t1, t2 in merges:
            f.write(t1.decode("latin-1") + " " + t2.decode("latin-1") + "\n")

    print(f"Saved vocab → {vocab_path}")
    print(f"Saved merges → {merges_path}")

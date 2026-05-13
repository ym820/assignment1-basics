# profile_bpe.py
import cProfile
import os
import pstats
import json
import tracemalloc
from tests.adapters import run_train_bpe
from tests.sample_file import sample_file

OUTPUT_DIR = "outputs/bpe"
os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == '__main__':
    tracemalloc.start()
    profiler = cProfile.Profile()
    profiler.enable()

    file_path = "data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 1000
    is_debug = False
    if is_debug:
        input_path = "data/debug_sample.txt"
        sample_file(file_path, input_path, 1_000_000_000)
    else:
        input_path = file_path

    print(f"Size of input file: {os.path.getsize(input_path) / 1024 / 1024:.2f} MB")

    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
    )

    file_name = os.path.basename(input_path).split(".")[0]
    with open(f"{OUTPUT_DIR}/bpe_vocab_{file_name}_{len(vocab)}.json", "w") as f:
        json.dump({str(k): v.decode('latin-1') for k, v in vocab.items()}, f, indent=2)
    print(f"Token with longest byte length: {max(vocab.values(), key=len)}")

    profiler.disable()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Current memory: {current / 1024 / 1024:.2f} MB")
    print(f"Peak memory:    {peak / 1024 / 1024:.2f} MB")

    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(20)  # top 20 slowest functions
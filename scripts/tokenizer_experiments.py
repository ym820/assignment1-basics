import os
import argparse
import random
import mmap
import time
import numpy as np
from cs336_basics.tokenizer import Tokenizer

EOS = "<|endoftext|>"
PATHS = {
    "tinystories": {
        "train": "data/TinyStoriesV2-GPT4-train.txt",
        "valid": "data/TinyStoriesV2-GPT4-valid.txt",
        "vocab": "outputs/bpe/bpe_vocab_tinystories_10000_train_bpe.pkl",
        "merges": "outputs/bpe/bpe_merges_tinystories_10000_train_bpe.pkl"
    },
    "owt": {
        "train": "data/owt_train.txt",
        "valid": "data/owt_valid.txt",
        "vocab": "outputs/bpe/bpe_vocab_owt_32000_train_bpe.pkl",
        "merges": "outputs/bpe/bpe_merges_owt_32000_train_bpe.pkl"
    }
}

def sample_with_random_offset(path: str, count:int=10, size=None):
    docs: list[str] = []
    seen_pair = set()
    seen_counter = 0
    sample_size = 0
    file_size = os.path.getsize(path)
    if size is not None:
        count = float('inf')
    else:
        size = file_size

    with open(path, "r+b") as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            EOS_bytes = EOS.encode("utf-8")
            while len(docs) < count and sample_size < size and seen_counter <= 1000:
                random_offset = random.randrange(0, file_size)

                index_before = mm.rfind(EOS_bytes, 0, random_offset)
                if index_before == -1:
                    index_before = 0
                else:
                    index_before += len(EOS_bytes)

                index_after = mm.find(EOS_bytes, max(0, random_offset - (len(EOS) - 1)), file_size)
                if index_after == -1:
                    index_after = file_size

                if (index_before, index_after) in seen_pair:
                    seen_counter += 1
                    continue
                else:
                    seen_pair.add((index_before, index_after))
                    docs.append(mm[index_before:index_after].decode("utf-8"))
                    sample_size += (index_after - index_before)
                    seen_counter = 0

    return docs

def get_compression_ratio(samples:list[str], tokenizer: Tokenizer):
    total_bytes = 0
    total_tokens = 0

    for sample in samples:
        total_tokens += len(tokenizer.encode(sample))
        total_bytes += len(sample.encode("utf-8"))

    compression_ratio = total_bytes / total_tokens
    return compression_ratio

def get_throughput(samples:list[str], tokenizer: Tokenizer):
    total_bytes = 0
    total_time = 0

    for sample in samples:
        start_time = time.perf_counter()
        tokenizer.encode(sample)
        end_time = time.perf_counter()
        total_time += (end_time - start_time)
        total_bytes += len(sample.encode("utf-8"))

    return total_bytes / total_time

def encode_to_uint16(tokenizer:Tokenizer, input_path: str, output_path: str):
    buffer_size = 1_000_000
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(input_path, mode='r', encoding="utf-8", errors="ignore") as f_in, open(output_path, "wb") as f_out:
        print(f"Encoding {input_path} ({os.path.getsize(input_path):,} bytes) -> {output_path}")
        start = time.perf_counter()
        total = 0
        buffer = np.empty(buffer_size, dtype=np.uint16)
        n = 0

        for token_id in tokenizer.encode_iterable(f_in):
            buffer[n] = token_id
            n += 1

            if n == buffer_size:
                f_out.write(buffer.tobytes())
                total += n
                elapsed = time.perf_counter() - start
                print(f"  {total:,} tokens | {elapsed:.0f}s | {total / elapsed:,.0f} tokens/s", flush=True)
                n = 0
        
        if n:
            f_out.write(buffer[:n].tobytes())

        total += n
        elapsed = time.perf_counter() - start
        print(f"Done: {total:,} tokens in {elapsed:.1f}s | "
              f"{os.path.getsize(input_path) / total:.2f} bytes/token")

def test(path, tokenizer:Tokenizer):
    token_ids = np.memmap(path, dtype=np.uint16, mode='r')
    if "tinystories" in path:
        assert token_ids.max() < 10000
        decoded_text = tokenizer.decode(token_ids[:50].tolist())
        print("Decoded text:")
        print(decoded_text)
    
        with open('data/TinyStoriesV2-GPT4-valid.txt', 'r') as f:
            text = f.read(200)
        print(text)
    else:
        assert token_ids.max() < 32000
        decoded_text = tokenizer.decode(token_ids[:50].tolist())
        print("Decoded text:")
        print(decoded_text)
    
        with open('data/owt_valid.txt', 'r') as f:
            text = f.read(200)
        print(text)
        
if __name__ == "__main__":
    ts_dataset_path = PATHS.get("tinystories").get("train")
    ts_vocab_path = PATHS.get("tinystories").get("vocab")
    ts_merges_path = PATHS.get("tinystories").get("merges")

    owt_dataset_path = PATHS.get("owt").get("train")
    owt_vocab_path = PATHS.get("owt").get("vocab")
    owt_merges_path = PATHS.get("owt").get("merges")

    ts_tokenizer = Tokenizer.from_files(ts_vocab_path, ts_merges_path, [EOS])
    owt_tokenizer = Tokenizer.from_files(owt_vocab_path, owt_merges_path, [EOS])

    p = argparse.ArgumentParser()
    p.add_argument("--encode", action="store_true", help="Encode")
    args = p.parse_args()

    if not args.encode:

        # (a) Tokenizer compression ratio for tinystories and owt
        ts_samples = sample_with_random_offset(ts_dataset_path)
        owt_samples = sample_with_random_offset(owt_dataset_path)

        tinystories_ratio = get_compression_ratio(ts_samples, ts_tokenizer)
        owt_ratio = get_compression_ratio(owt_samples, owt_tokenizer)

        # (b) Cross-domain compression ratio
        ts_on_owt = get_compression_ratio(owt_samples, ts_tokenizer)
        owt_on_ts = get_compression_ratio(ts_samples, owt_tokenizer)

        print("========Tokenizer Compression Ratio========")
        print(f"Tinystories: {tinystories_ratio} bytes/token")
        print(f"OWT: {owt_ratio} bytes/token")
        print(f"Tinystories tokenizer on OWT data: {ts_on_owt} bytes/token")
        print(f"OWT tokenizer on Tinystories data: {owt_on_ts} bytes/token")


        # (c) Throughput
        ts_samples_5mb = sample_with_random_offset(ts_dataset_path, size=5_000_000)
        owt_samples_5mb = sample_with_random_offset(owt_dataset_path, size=5_000_000)

        ts_throughput = get_throughput(ts_samples_5mb, ts_tokenizer)
        ts_estimated_days = 825e9 / ts_throughput / 60 / 60 / 24
        owt_throughput = get_throughput(owt_samples_5mb, owt_tokenizer)
        owt_estimated_days = 825e9 / owt_throughput / 60 / 60 / 24

        print("========Tokenizer Throughput========")
        print(f"Tinystories: {ts_throughput} bytes/second")
        print(f"OWT: {owt_throughput} bytes/second")
        print("Estimated time to tokenize 825 GB text:")
        print(f"Tinystories: {ts_estimated_days} days")
        print(f"OWT: {owt_estimated_days} days")

    else:
        # (d) Encode TinyStories and OWT

        encode_to_uint16(ts_tokenizer, PATHS.get("tinystories").get("train"), "outputs/training/tinystories_train.uint16.bin")
        # encode_to_uint16(ts_tokenizer, PATHS.get("tinystories").get("valid"), "outputs/training/tinystories_valid.uint16.bin")
        # test("outputs/training/tinystories_valid.uint16.bin", ts_tokenizer)

        encode_to_uint16(owt_tokenizer, PATHS.get("owt").get("train"), "outputs/training/owt_train.uint16.bin")
        # encode_to_uint16(owt_tokenizer, PATHS.get("owt").get("valid"), "outputs/training/owt_valid.uint16.bin")
        # test("outputs/training/owt_valid.uint16.bin", owt_tokenizer)
    




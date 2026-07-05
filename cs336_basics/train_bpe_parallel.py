import os
from collections import Counter
from multiprocessing import Pool
from cs336_basics.pretokenization_example import find_chunk_boundaries

GPT2_PRETOKENIZATION_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT2_REGEX = None

# ------- Pre-tokenization Optimization (Parallelism) Helper Functions -------
def init_rx():
    global GPT2_REGEX
    import regex as re
    GPT2_REGEX = re.compile(GPT2_PRETOKENIZATION_PATTERN)

def get_word_freq(text: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    if not text:
        return {}
    
    # split by each special token and drop the special tokens
    spans = [text]
    for special_token in special_tokens:
        new_spans: list[str] = []
        for span in spans:
            if span:
                new_spans.extend(span.split(special_token))
        spans = new_spans
    
    word_freq: dict[tuple[bytes, ...], int] = {}
    for span in spans:
        if not span:
            continue
        for m in GPT2_REGEX.finditer(span):
            match_text = m.group(0)
            if not match_text:
                continue
            token_bytes = match_text.encode("utf-8")
            word = tuple(bytes([b]) for b in token_bytes)
            word_freq[word] = word_freq.get(word, 0) + 1
    
    return word_freq

def process_chunk(args) -> dict[tuple[bytes, ...], int]:
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)
    
    text = chunk.decode("utf-8", errors="ignore")
    return get_word_freq(text, special_tokens)

def build_word_freq_serial(input_path: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    init_rx()
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    return get_word_freq(text, special_tokens)

def build_word_freq_parallel(
    input_path: str,
    special_tokens: list[str],
    num_processes: int,
    num_chunks: int = -1
) -> dict[tuple[bytes, ...], int]:
    
    if num_chunks < 0:
        num_chunks = max(num_processes * 32, num_processes)

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0].encode("utf-8"))

    tasks = [(str(input_path), s, e, special_tokens) for s, e in zip(boundaries[:-1], boundaries[1:])]

    merged = Counter()
    with Pool(processes=num_processes, initializer=init_rx) as pool:
        for partial in pool.imap_unordered(process_chunk, tasks, chunksize=1):
            merged.update(partial)
    
    return dict(merged)

# ------- Merge Optimization Helper Functions -------
def get_pair_counts(word: tuple[bytes, ...]) -> dict[tuple[bytes, bytes], int]:
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    if len(word) < 2:
        return pair_counts
    prev = word[0]
    for cur in word[1:]:
        pair = (prev, cur)
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        prev = cur
    return pair_counts

def build_pair_helpers(
    word_freq: dict[tuple[bytes, ...], int]
)-> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]]:
    """
    Build:
      - pair_counts: global weighted counts for each adjacent pair
      - pair_to_words: inverted index (pair -> set of words containing that pair)
    """ 
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}

    for word, freq in word_freq.items():
        if len(word) < 2:
            continue
        current_pair_counts = get_pair_counts(word)
        
        for pair, count in current_pair_counts.items():
            pair_counts[pair] = pair_counts.get(pair, 0) + count * freq
            word_set = pair_to_words.get(pair)
            if word_set is None:
                pair_to_words[pair] = {word}
            else:
                word_set.add(word)
    
    return pair_counts, pair_to_words

def remove_word(
    word: tuple[bytes, ...],
    freq: int,
    pair_counts: tuple[dict[tuple[bytes, bytes], int]],
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]
):
    current_pair_counts = get_pair_counts(word)
    for pair, count in current_pair_counts.items():
        word_set = pair_to_words.get(pair)
        if word_set is not None:
            word_set.discard(word)
            if not word_set:
                del pair_to_words[pair]
        
        new_count = pair_counts.get(pair, 0) - count * freq
        if new_count <= 0:
            pair_counts.pop(pair, None)
        else:
            pair_counts[pair] = new_count

def add_word(
    word: tuple[bytes, ...],
    freq: int,
    pair_counts: tuple[dict[tuple[bytes, bytes], int]],
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
    is_new: bool
):
    current_pair_counts = get_pair_counts(word)
    for pair, count in current_pair_counts.items():
        pair_counts[pair] = pair_counts.get(pair, 0) + count * freq
        if is_new:
            word_set = pair_to_words.get(pair)
            if word_set is None:
                pair_to_words[pair] = {word}
            else:
                word_set.add(word)


# ------- Main train_bpe Function -------

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # --- Step 1: Initialize vocabulary ---
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")


    # --- Step 2: Pre-tokenization and counting ---
    num_processes = min(8, os.cpu_count() or 1)
    print(f"Pre-tokenizing with {num_processes} processes...")
    if num_processes <= 1:
        word_freq = build_word_freq_serial(input_path, special_tokens)
    else:
        word_freq = build_word_freq_parallel(input_path, special_tokens, num_processes)
    
    if not word_freq:
        return vocab, []
    

    # --- Step 3: BPE merges ---
    merges: list[tuple[bytes, bytes]] = []
    pair_counts, pair_to_words = build_pair_helpers(word_freq)
    next_id = len(vocab)
    while next_id < vocab_size:
        if not pair_counts:
            break

        # choose most frequent; tie-break by lexicographically largest pair
        (a, b), best_count = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
        if best_count <= 0:
            break

        new_token = a + b
        merges.append((a, b))
        vocab[next_id] = new_token
        next_id += 1

        # Find impacted set of word if merged a, b
        impacted_word_set = pair_to_words.get((a, b))
        if not impacted_word_set:
            pair_counts.pop((a, b), None)
            continue
        
        word_with_new_token_freq: dict[tuple[bytes, ...], int] = {}
        for word in list(impacted_word_set):
            freq = word_freq.get(word)
            if freq is None:
                continue
            
            # Update pair helpers because we want to remove token a and b
            remove_word(word, freq, pair_counts, pair_to_words)
            del word_freq[word]
            
            # Update pair helpers because we want to add new token

            # First, get the word with new token
            # Replace occurences of consecutive a and b with new_token in word
            word_seq_with_new_token: list[bytes] = []
            i = 0
            word_len = len(word)
            while i < word_len:
                if i < word_len - 1 and word[i] == a and word[i+1] == b:
                    word_seq_with_new_token.append(new_token)
                    i += 2
                else:
                    word_seq_with_new_token.append(word[i])
                    i += 1
            word_with_new_token = tuple(word_seq_with_new_token)

            # Track the freq to be added back to the word_freq
            word_with_new_token_freq[word_with_new_token] = word_with_new_token_freq.get(word_with_new_token, 0) + freq
        
        for word_with_new_token, add_freq in word_with_new_token_freq.items():
            is_new = word_with_new_token not in word_freq
            word_freq[word_with_new_token] = word_freq.get(word_with_new_token, 0) + add_freq
            add_word(word_with_new_token, add_freq, pair_counts, pair_to_words, is_new)
    
    return vocab, merges

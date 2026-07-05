import os

GPT2_PRETOKENIZATION_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT2_REGEX = None

def get_gpt2_regex():
    global GPT2_REGEX
    if GPT2_REGEX is None:
        import regex as re

        GPT2_REGEX = re.compile(GPT2_PRETOKENIZATION_PATTERN)
    return GPT2_REGEX

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
    gpt2_regex = get_gpt2_regex()
    word_freq: dict[tuple[bytes, ...], int] = {}

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    if not text:
        return vocab, []
    
    # split by each special token and drop the special tokens
    spans = [text]
    for special_token in special_tokens:
        new_spans: list[str] = []
        for span in spans:
            if span:
                new_spans.extend(span.split(special_token))
        spans = new_spans
    
    for span in spans:
        if not span:
            continue
        for m in gpt2_regex.finditer(span):
            match_text = m.group(0)
            if not match_text:
                continue
            token_bytes = match_text.encode("utf-8")
            word = tuple(bytes([b]) for b in token_bytes)
            word_freq[word] = word_freq.get(word, 0) + 1

    # --- Step 3: BPE merges ---
    merges: list[tuple[bytes, bytes]] = []
    next_id = len(vocab)
    while next_id < vocab_size:

        pair_counts: dict[tuple[bytes, bytes], int] = {}
        for word, freq in word_freq.items():
            if len(word) < 2:
                continue
            prev = word[0]
            for cur in word[1:]:
                pair = (prev, cur)
                pair_counts[pair] = pair_counts.get(pair, 0) + freq
                prev = cur
        
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

        # replace occurences of (a, b) in every word
        new_word_freq: dict[tuple[bytes, ...], int] = {}
        for word, freq in word_freq.items():
            if len(word) < 2:
                new_word_freq[word] = new_word_freq.get(word, 0) + freq
                continue

            merged: list[bytes] = []
            i = 0
            word_len = len(word)
            while i < word_len:
                if i < word_len - 1 and word[i] == a and word[i+1] == b:
                    merged.append(new_token)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            
            merged_word = tuple(merged)
            new_word_freq[merged_word] = new_word_freq.get(merged_word, 0) + freq
        
        word_freq = new_word_freq
    
    return vocab, merges

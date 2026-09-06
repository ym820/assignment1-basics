"""
Three terms are used frequently in this script:
token  : bytes                    # b'l', or b'ow' after a merge
word   : tuple[bytes, ...]        # a sequence of tokens, e.g. (b'l', b'o', b'w') and (b'l', b'ow')
pair   : tuple[bytes, bytes]      # two adjacent tokens in a word
"""
# Imports
import os
import heapq
import regex as re
from collections import defaultdict
from multiprocessing import get_context
from cs336_basics.pretokenization_example import find_chunk_boundaries

GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class _ReverseKey:
    """Wrapper that inverts comparison for use inside a min-heap tuple.

    Why we need this:
      heapq is a MIN-heap, so we store (-count, pair) to get the MAX count
      popped first. But when two pairs have the same count, heapq compares
      the next element of the tuple (the pair itself). Python's default
      bytes comparison would then break ties toward the LEXICOGRAPHICALLY
      SMALLEST pair — but the BPE reference implementation (and the tests)
      expect ties broken toward the LARGEST pair. So we wrap the pair in
      _ReverseKey, which flips <, <=, so that "smallest in heap order"
      corresponds to "lexicographically largest" bytes tuple.

      Example: pairs (b'a', b'b') and (b'x', b'y') both have count 5.
        Without _ReverseKey: (b'a', b'b') pops first (it's "smaller").
        With    _ReverseKey: (b'x', b'y') pops first (tie → pick larger).
    """
    __slots__ = ('key',)
    def __init__(self, key): self.key = key
    def __eq__(self, other): return self.key == other.key
    def __lt__(self, other): return self.key > other.key   # inverted
    def __le__(self, other): return self.key >= other.key  # inverted


# ---------- Pre-tokenization ----------

def _count_pretokens(text: str, st_pattern: str) -> dict[bytes, int]:
    """Run GPT-2 pre-tokenization on `text` and return {pre_token_bytes: count}.

    Two-stage split:
      1. First, split OUT any special tokens (e.g. "<|endoftext|>") using
         st_pattern. Specials must never be fed into GPT2_PATTERN because GPT2_PATTERN might
         break them up into punctuation + letters + punctuation.
         NOTE: re.split DROPS the matched delimiter, so specials are
         discarded here entirely — they are re-added to the vocab separately
         in train_bpe and don't participate in merges at all.
      2. Then, on each non-special segment, run the GPT2_PATTERN regex to extract
         pre-tokens.

    Returns keys as raw `bytes` (not tuples of single-byte bytes). Why?
    Because this function runs inside worker processes, and `bytes` is more
    compact to pickle/ship across process boundaries than a tuple-of-bytes.
    The caller converts to tuple form later, in the main process.

    Example:
      text = "hi hi!"
      GPT2_PATTERN matches: ["hi", " hi", "!"]
      returns: {b"hi": 1, b" hi": 1, b"!": 1}
    """
    counts: dict[bytes, int] = {}
    # `st_pattern` can be empty string if there are no specials — in that
    # case re.split would still return [text], but we short-circuit to avoid
    # compiling an empty pattern.
    parts = re.split(st_pattern, text) if st_pattern else [text]
    for part in parts:
        for match in re.finditer(GPT2_PATTERN, part):
            word_bytes = match.group().encode("utf-8")
            counts[word_bytes] = counts.get(word_bytes, 0) + 1
    return counts


def _process_chunk(args):
    """Worker function for the multiprocessing pool.

    Arguments come in as a single tuple because pool.map only passes one arg.
    We open the file in each worker rather than passing file contents through
    the pool — this lets each worker read its own byte range directly from
    disk in parallel, and avoids serializing huge strings through pickle.

    `errors="ignore"` drops any bytes that aren't valid UTF-8. This can
    happen at chunk boundaries if we accidentally cut a multi-byte codepoint
    in half (though find_chunk_boundaries tries to avoid this).
    """
    input_path, start, end, st_pattern = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return _count_pretokens(chunk, st_pattern)


def _get_pre_token_counts_parallel(
    input_path: str, num_processes: int, st_pattern: str
) -> dict[tuple[bytes, ...], int]:
    """Parallel pre-tokenization using a process pool.

    Pipeline:
      1. Compute byte-range boundaries so each worker gets a non-overlapping
         chunk. `find_chunk_boundaries` snaps the cuts to <|endoftext|>
         markers so we never split a document in half.
      2. Spawn workers; each returns a partial {bytes: count} dict.
      3. Reduce: merge all partials into a single dict, converting each
         word from `bytes` → `tuple[bytes, ...]` (the format the merge loop
         wants, where each element is one single-byte bytes object).

    Why convert in the reducer instead of in the workers?
    Two reasons:
      - `bytes` is ~40% cheaper to pickle than `tuple[bytes, ...]` of the
        same content, so workers return less data.
      - The conversion is O(total characters), which is fast in pure Python
        and only runs once per unique word in the main process.
    """
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    chunk_args = [
        (input_path, start, end, st_pattern)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    # get_context("spawn") forces a fresh Python interpreter per worker,
    # which is safer on macOS and avoids inheriting random module state.
    with get_context("spawn").Pool(processes=num_processes) as pool:
        chunk_results = pool.map(_process_chunk, chunk_args)

    pre_token_counts: dict[tuple[bytes, ...], int] = defaultdict(int)
    for chunk_counts in chunk_results:
        for word_bytes, count in chunk_counts.items():
            # word_bytes[i:i+1] returns a length-1 `bytes` (hashable, usable
            # as a dict key). word_bytes[i] alone would return an int.
            # Example: b"hi" → (b"h", b"i")
            key = tuple(word_bytes[i:i + 1] for i in range(len(word_bytes)))
            pre_token_counts[key] += count
    return pre_token_counts


def _get_pre_token_counts_serial(
    input_path: str, st_pattern: str
) -> dict[tuple[bytes, ...], int]:
    """Single-process fallback. Same output shape as the parallel version."""
    with open(input_path, "rb") as f:
        text = f.read().decode("utf-8", errors="ignore")
    byte_counts = _count_pretokens(text, st_pattern)
    return {
        tuple(word_bytes[i:i + 1] for i in range(len(word_bytes))): count
        for word_bytes, count in byte_counts.items()
    }


# ---------- Heap helpers (lazy deletion) ----------
#
# Why a heap at all? Each merge iteration we need the highest-count pair.
# With ~100k unique pairs in a real corpus and ~30k merges, doing
# `max(pair_counts, key=...)` would be ~3 billion comparisons. A heap
# amortizes this to O(log n) per operation.
#
# Why LAZY deletion? When we apply a merge, many pair counts change.
# Updating a heap entry in place is O(n). Instead we just push a NEW entry
# with the updated count, and when we pop we check whether the stored count
# still matches pair_counts — if not, the entry is stale and we discard it.
# This means the heap can grow larger than `len(pair_counts)` over time,
# but each entry is only ever popped once, so total work is still bounded.


def _push_pair(heap: list, pair: tuple[bytes, bytes], count: int) -> None:
    """Push a (pair, count) entry onto the max-heap (stored as -count)."""
    heapq.heappush(heap, (-count, _ReverseKey(pair)))


def _pop_best_pair(
    heap: list, pair_counts: dict[tuple[bytes, bytes], int]
) -> tuple[bytes, bytes]:
    """Pop the highest-count pair, skipping stale entries.

    Example of staleness:
      Suppose (b'e', b's') had count 10, we pushed (-10, key) into the heap.
      Then a merge decremented it to 7, and we pushed (-7, key).
      When we later pop (-10, key): pair_counts[(b'e',b's')] == 7 ≠ 10,
      so this entry is stale — discard it and pop again.
      Next pop gives (-7, key): 7 == 7 → fresh, return it.
    """
    while True:
        neg_count, rk = heapq.heappop(heap)
        pair = rk.key
        if pair_counts.get(pair, 0) == -neg_count:
            return pair


def _shift_pair(
    old_pair: tuple[bytes, bytes],
    new_pair: tuple[bytes, bytes],
    count: int,
    pair_counts: dict[tuple[bytes, bytes], int],
    heap: list,
) -> None:
    """Decrement `old_pair` and increment `new_pair` by `count` each.

    Called every time a merge replaces one adjacent pair with a new one in
    some word. `count` is the word's frequency in the corpus — if the word
    appears 5 times, the pair appears 5 times too (once per word copy), so
    we shift by 5, not 1.

    Only pushes fresh heap entries for pairs whose new count is still > 0.
    (We don't bother pushing old_pair if its count dropped to 0 — lazy
    deletion means the stale positive entry still in the heap will be
    discarded naturally next time we pop it.)
    """
    pair_counts[old_pair] -= count
    if pair_counts[old_pair] > 0:
        _push_pair(heap, old_pair, pair_counts[old_pair])
    pair_counts[new_pair] += count
    _push_pair(heap, new_pair, pair_counts[new_pair])


# ---------- Merge application ----------

def _merge_in_word(
    wid: int,
    token1: bytes,
    token2: bytes,
    new_token: bytes,
    word_id_to_word: dict[int, tuple[bytes, ...]],
    word_id_to_count: dict[int, int],
    pair_counts: dict[tuple[bytes, bytes], int],
    pair_to_word_ids: dict[tuple[bytes, bytes], set],
    heap: list,
) -> None:
    """Apply merge (token1, token2) → new_token to a single word.

    This is where all the subtle index-maintenance lives. Four steps:
      1. Scan the word to find every position where the pair occurs.
      2. Build new_word by replacing those positions with new_token.
      3. Update word_id_to_word (the word ID stays the same).
      4a. For every pair whose COUNT changed (the neighbors of each merge
          site), shift pair_counts and push new heap entries.
      4b. For every pair whose MEMBERSHIP in this word changed, update
          pair_to_word_ids using set difference.

    Steps 4a and 4b are separate because counts and membership behave
    differently — see the comments on each for why.
    """
    old_word = word_id_to_word[wid]
    count = word_id_to_count[wid]

    # --- Step 1: find all merge positions ---
    #
    # Critical: jump by 2 on a match. If we jumped by 1 we'd double-count
    # overlapping matches, which matters for patterns like ABAB with pair AB.
    #
    # Non-overlapping example:
    #   old_word = (a, b, c, a, b), pair = (a, b)
    #   i=0 match  → positions=[0], i=2
    #   i=2 (c,a) no match → i=3
    #   i=3 match  → positions=[0, 3], i=5
    #
    # Adjacent example (this is the tricky case):
    #   old_word = (a, b, a, b), pair = (a, b)
    #   i=0 match  → positions=[0], i=2
    #   i=2 match  → positions=[0, 2], i=4
    #   Result: two merges at distance 2 ("adjacent" in the merge sense).
    positions = []
    i = 0
    while i < len(old_word) - 1:
        if old_word[i] == token1 and old_word[i + 1] == token2:
            positions.append(i)
            i += 2
        else:
            i += 1
    pos_set = set(positions)

    # --- Step 2: build new_word ---
    #
    # Walk old_word once, emitting new_token at each merge position and
    # plain tokens elsewhere.
    #
    # Example: old_word=(a,b,c,a,b), positions={0,3}, new_token=ab
    #   i=0 in pos_set → emit ab, i=2
    #   i=2 not in pos_set → emit c, i=3
    #   i=3 in pos_set → emit ab, i=5
    #   new_word = (ab, c, ab)
    new_word_list = []
    i = 0
    while i < len(old_word):
        if i in pos_set:
            new_word_list.append(new_token)
            i += 2
        else:
            new_word_list.append(old_word[i])
            i += 1
    new_word = tuple(new_word_list)

    # --- Step 3: update word_id_to_word in place ---
    #
    # The word's ID doesn't change. This is the whole point of using stable
    # integer IDs: any pair_to_word_ids entries that still reference this word
    # (because their pair was unchanged by this merge) keep working without
    # modification — they still point at the same ID, which now resolves to
    # new_word.
    word_id_to_word[wid] = new_word

    # --- Step 4a: shift pair_counts for the pairs whose counts changed ---
    #
    # Key insight: the only pairs whose counts actually change are the ones
    # immediately adjacent to a merge site. Pairs elsewhere in the word are
    # entirely unchanged — same tokens on both sides, same count.
    #
    # For a merge at position p in old_word, the affected pairs are:
    #   LEFT neighbor:  (old_word[p-1], token1)  →  (old_word[p-1], new_token)
    #   RIGHT neighbor: (token2, old_word[p+2])  →  (new_token, old_word[p+2])
    #
    # ~~~ The adjacent-merge special case ~~~
    # When two merges happen at distance 2 (e.g. positions [0, 2] in (a,b,a,b)):
    #   old_word = (A, B, A, B)
    #   new_word = (AB, AB)
    # The pair (B, A) that sat BETWEEN the two merges needs to become (AB, AB).
    # We account for this as the RIGHT NEIGHBOR of position 0 (detecting that
    # p+2 is also in pos_set) and SKIP the left neighbor of position 2 (since
    # that would try to process the same seam a second time, double-counting).
    #
    # Concrete example (normal): old_word=(x,a,b,y), merging (a,b) at p=1
    #   Left:  (x,a) → (x,ab)
    #   Right: (b,y) → (ab,y)
    #
    # Concrete example (adjacent): old_word=(a,b,a,b), positions=[0,2]
    #   p=0 right: p+2=2 in pos_set → seam (b,a) → (ab,ab)
    #   p=2 left:  p-2=0 in pos_set → SKIP (already handled)
    for p in positions:
        # Left neighbor (skip if the previous position is also a merge site)
        if p > 0 and (p - 2) not in pos_set:
            left_tok = old_word[p - 1]
            _shift_pair((left_tok, token1), (left_tok, new_token),
                        count, pair_counts, heap)

        # Right neighbor
        if p + 2 < len(old_word):
            if (p + 2) in pos_set:
                # Adjacent merge: the seam (token2, token1) becomes
                # (new_token, new_token). Both sides are being merged.
                _shift_pair((token2, token1), (new_token, new_token),
                            count, pair_counts, heap)
            else:
                right_tok = old_word[p + 2]
                _shift_pair((token2, right_tok), (new_token, right_tok),
                            count, pair_counts, heap)

    # --- Step 4b: update pair_to_word_ids membership via set difference ---
    #
    # pair_to_word_ids[P] answers "which word IDs contain pair P?" — it tracks
    # MEMBERSHIP, not COUNT. A word either contains a pair or it doesn't.
    #
    # Why not update pair_to_word_ids inside the Step 4a loop? Because of this bug:
    #
    #   old_word = (a, b, c, a, b), merging (b, c) at position 1
    #   Neighbor loop would see (a, b) as the left neighbor and want to
    #   remove wid from pair_to_word_ids[(a, b)] — but (a, b) STILL EXISTS in
    #   new_word at position 3! Removing wid there would be wrong.
    #
    # Set difference sidesteps the whole issue: we compute the set of pairs
    # that were in old_word, the set that are in new_word, and only modify
    # pair_to_word_ids for pairs that genuinely appeared or disappeared.
    #
    # Example: old_word=(a,b,c,a,b), new_word=(a,bc,a,b)
    #   old_pairs = {(a,b), (b,c), (c,a)}
    #   new_pairs = {(a,bc), (bc,a), (a,b)}
    #   removed  = {(b,c), (c,a)}         ← discard wid from these
    #   added    = {(a,bc), (bc,a)}       ← add wid to these
    #   (a,b) is in both → no change (correct!)
    old_pair_set = set(zip(old_word, old_word[1:]))
    new_pair_set = set(zip(new_word, new_word[1:]))
    for pair in old_pair_set - new_pair_set:
        pair_to_word_ids[pair].discard(wid)
    for pair in new_pair_set - old_pair_set:
        pair_to_word_ids[pair].add(wid)


def _apply_merge(
    token1: bytes,
    token2: bytes,
    new_token: bytes,
    word_id_to_word: dict[int, tuple[bytes, ...]],
    word_id_to_count: dict[int, int],
    pair_counts: dict[tuple[bytes, bytes], int],
    pair_to_word_ids: dict[tuple[bytes, bytes], set],
    heap: list,
) -> None:
    """Apply one BPE merge across every word that contains the pair.

    pair_to_word_ids[(token1, token2)] is our index of "which words need updating."
    Without it we'd have to scan every word every merge, which would turn
    this into an O(V * W) algorithm. With it, we only touch the words that
    actually contain the pair being merged.

    The list() snapshot is required because _merge_in_word mutates pair_to_word_ids
    during iteration — specifically, it may add wid to OTHER pair_to_word_ids
    entries via set difference, and in pathological cases also discard from
    pair_to_word_ids[(token1, token2)] itself if the word happened to contain the
    merged pair more than once.
    """
    for wid in list(pair_to_word_ids[(token1, token2)]):
        _merge_in_word(
            wid, token1, token2, new_token,
            word_id_to_word, word_id_to_count,
            pair_counts, pair_to_word_ids, heap,
        )
    # After applying to every affected word, the merged pair is guaranteed
    # to no longer exist anywhere in the corpus. Drop it from both indices.
    del pair_counts[(token1, token2)]
    del pair_to_word_ids[(token1, token2)]


# ---------- Top-level training entry point ----------

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the given input file.

    High-level flow:
      1. Initialize vocab with 256 single-byte tokens + any special tokens.
      2. Pre-tokenize the corpus into a {word: count} dict. Words are
         tuples of single-byte bytes objects (the BPE "starting state").
      3. Build indices: pair_counts (pair → total count) and pair_to_word_ids
         (pair → set of word IDs containing it).
      4. Build a max-heap of pair_counts with lazy deletion.
      5. Until vocab reaches target size:
           - Pop the best pair from the heap.
           - Record the merge, add new_token to vocab.
           - Apply the merge to every affected word, incrementally updating
             pair_counts, pair_to_word_ids, and the heap.

    Args:
        input_path:    Path to the input text file for training.
        vocab_size:    Desired final vocabulary size (includes single-byte
                       tokens AND special tokens — so if you pass 1000 with
                       1 special token, you'll end up with 256 + 1 + 743
                       merged tokens = 743 merge rules).
        special_tokens: Special token strings to reserve in the vocabulary.
                        These are split OUT of the text before pre-tokenization
                        so they never participate in merges.

    Returns:
        vocab:  id → token bytes
        merges: ordered list of (token1, token2) merge rules. Order matters
                for the Tokenizer — merge i was learned before merge i+1 and
                must be applied at higher priority.
    """
    # --- Step 1: Initialize vocabulary ---
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")
    merges: list[tuple[bytes, bytes]] = []


    # --- Step 2: Pre-tokenization ---
    num_processes = min(8, os.cpu_count() or 1)
    st_pattern = "|".join(re.escape(tok) for tok in special_tokens)
    print(f"Pre-tokenizing with {num_processes} processes...")
    if num_processes > 1:
        pre_token_counts = _get_pre_token_counts_parallel(
            input_path, num_processes, st_pattern
        )
    else:
        pre_token_counts = _get_pre_token_counts_serial(input_path, st_pattern)


    # --- Step 3: Build BPE indices ---
    #
    # Assign each unique pre-token a stable integer ID. This is a key design
    # choice: it lets pair_to_word_ids use cheap int sets (not tuple-of-bytes sets),
    # and it lets us update a word's token sequence IN PLACE without having
    # to touch pair_to_word_ids entries for unchanged pairs — they still point at
    # the same ID, which now resolves to the updated word.
    #
    # Example: pre_token_counts = {(b'l',b'o',b'w'): 3, (b'h',b'i'): 5}
    #   word_id_to_word = {0: (b'l',b'o',b'w'), 1: (b'h',b'i')}
    #   word_id_to_count     = {0: 3,                 1: 5}
    word_id_to_word: dict[int, tuple[bytes, ...]] = {}
    word_id_to_count: dict[int, int] = {}
    for wid, (word, count) in enumerate(pre_token_counts.items()):
        word_id_to_word[wid] = word
        word_id_to_count[wid] = count

    # pair_counts[P] = total corpus frequency of adjacent pair P,
    #                  summed over all words (each weighted by word_id_to_count).
    # pair_to_word_ids[P]   = set of word IDs whose current token sequence contains P.
    #
    # Example (continuing the above): word_id=0 is (b'l',b'o',b'w') count=3
    #   pair_counts[(b'l',b'o')] += 3,  pair_to_word_ids[(b'l',b'o')].add(0)
    #   pair_counts[(b'o',b'w')] += 3,  pair_to_word_ids[(b'o',b'w')].add(0)
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_word_ids: dict[tuple[bytes, bytes], set] = defaultdict(set)
    for wid, word in word_id_to_word.items():
        count = word_id_to_count[wid]
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += count
            pair_to_word_ids[pair].add(wid)


    # --- Step 4: Build heap ---
    # heapify is O(n), which is much faster than n individual pushes (O(n log n)).
    # See the "Heap helpers" section above for the lazy-deletion design.
    heap: list = [(-count, _ReverseKey(pair)) for pair, count in pair_counts.items()]
    heapq.heapify(heap)


    # --- Step 5: Merge loop ---
    # Each iteration: find best pair → record merge → apply merge to corpus.
    # Stop when we've created enough merged tokens to hit the target vocab size.
    while len(vocab) < vocab_size:
        token1, token2 = _pop_best_pair(heap, pair_counts)
        merges.append((token1, token2))
        new_token = token1 + token2        # bytes concatenation: b'e'+b's' → b'es'
        vocab[len(vocab)] = new_token
        _apply_merge(
            token1, token2, new_token,
            word_id_to_word, word_id_to_count,
            pair_counts, pair_to_word_ids, heap,
        )

    return vocab, merges

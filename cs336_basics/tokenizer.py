from typing import Iterable, Iterator
import regex as re
import pickle

# GPT-2 pre-tokenization regex. Splits text into "words" BEFORE BPE merges run.
# The goal: prevent merges from ever crossing these boundaries (e.g. stop "the" and
# " cat" from merging across the space, or "don" and "'t" from merging across the
# apostrophe contraction).
# Pattern pieces, in order:
#   'sdmtv... contractions: 's 'd 'm 't 'll 've 're  (kept attached to prev word)
#   ?\p{L}+    : optional leading space + unicode letters         (e.g. " cat")
#   ?\p{N}+    : optional leading space + unicode digits          (e.g. " 42")
#   ?[^\s\p{L}\p{N}]+ : optional leading space + run of punctuation
#   \s+(?!\S)  : trailing whitespace at end of string / before newline
#   \s+        : any other whitespace run
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    """Byte-level BPE tokenizer, compatible with GPT-2 style vocab + merges."""

    def __init__(self,
                 vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None
                 ):
        # vocab: id -> token bytes.  Every byte 0..255 is usually in here as a
        # single-byte token, plus all merged tokens created during training.
        self.vocab: dict[int, bytes] = vocab
        # merges: ordered list of pairs. Index = priority (lower index = applied first).
        # This order MUST match the order produced during training, otherwise
        # encoding will diverge from what the model was trained on.
        self.merges: list[tuple[bytes, bytes]] = merges

        # Reverse lookup: token bytes -> id. Built once so encode() is O(1) per token.
        self.byte_to_id: dict[bytes, int] = {b: i for i, b in self.vocab.items()}

        # merge_ranks: pair -> priority index. Used inside _bpe to find the
        # "best" (= lowest-rank = earliest-trained) mergeable pair in a word.
        # Dict lookup is O(1), much faster than scanning self.merges each step.
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {
            pair: idx for idx, pair in enumerate(self.merges)
        }

        # Pre-compile the pre-tokenization regex (expensive to compile, cheap to reuse).
        self.rx = re.compile(PAT)

        # ---- Special tokens ----
        # Specials (e.g. "<|endoftext|>") must NEVER be broken up by BPE or the
        # pre-tok regex. We handle them by splitting the input on specials first,
        # emitting their id directly, and only running BPE on the text between them.
        self.special_tokens: list[str] = special_tokens or []
        self.special_bytes: list[bytes] = []
        self.special_id: dict[str, int] = {}

        if self.special_tokens:
            # If a special token wasn't already in the trained vocab, append it
            # with a fresh id. NOTE: this mutates the caller's vocab dict.
            for st in self.special_tokens:
                encoded_st = st.encode("utf-8")
                if encoded_st not in self.byte_to_id:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = encoded_st
                    self.byte_to_id[encoded_st] = new_id
                self.special_id[st] = self.byte_to_id[encoded_st]
                self.special_bytes.append(encoded_st)

            # Sort longest-first so that if one special is a prefix of another
            # (e.g. "<|eot|>" vs "<|eot|><|eot|>"), the longer one matches first.
            # Regex alternation tries alternatives left-to-right, so order matters.
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            self._sepcial_re = re.compile("|".join(re.escape(s) for s in sorted_special_tokens))
            self._max_special_len = max(len(s) for s in self.special_tokens)
        else:
            self._sepcial_re = None
            self._max_special_len = 0

        # Cache: pre-token bytes -> list of merged token bytes.
        # Natural text reuses the same pre-tokens constantly (" the", " of", etc.),
        # so caching makes encoding dramatically faster. Unbounded — fine for this
        # assignment, but for huge corpora you'd want an LRU.
        self._bpe_cache: dict[bytes, list[bytes]] = {}

    @classmethod
    def from_files(cls,
                   vocab_filepath: str,
                   merges_filepath: str,
                   special_tokens: list[str] | None = None
                   ):
        """Load a tokenizer previously saved via pickle.dump of vocab and merges.
        NOTE: must be binary mode ('rb') because pickle is a binary format."""
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)

        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode a full string into a list of token ids.

        High-level flow:
          1. Split the text around special tokens (they pass through untouched).
          2. For each non-special segment: pre-tokenize with the GPT-2 regex.
          3. For each pre-token: apply BPE merges -> list of token bytes.
          4. Map each token bytes -> id via byte_to_id.
        """
        if not text:
            return []

        ids: list[int] = []

        # Fast path: no specials registered, just encode the whole string.
        if not self._sepcial_re:
            ids.extend(self._encode_plain(text))
            return ids

        # Walk through the text, alternating between "regular text" slices and
        # "special token" matches. `last` tracks where the previous special ended.
        last = 0
        for matched in self._sepcial_re.finditer(text):
            # Encode any plain text sitting BEFORE this special token.
            if matched.start() > last:
                ids.extend(self._encode_plain(text[last:matched.start()]))
            # Emit the special token's id directly — no pre-tok, no BPE.
            st = matched.group(0)
            ids.append(self.special_id[st])
            last = matched.end()
        # Tail: any plain text after the final special token.
        if last < len(text):
            ids.extend(self._encode_plain(text[last:]))

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Stream-encode an iterable of string chunks (e.g. lines from a huge file).

        Key idea: a chunk boundary may land in the middle of a pre-token (e.g.
        split " hello" into " hel" + "lo"), which would change the tokenization.
        So we buffer partial text, and only encode up to the START of the LAST
        regex match — that last match might still be incomplete, so we keep it
        in the buffer until the next chunk arrives (or until EOF).
        """
        buf = ""
        for chunk in iterable:
            buf += chunk
            matches = list(self.rx.finditer(buf))
            # Need at least 2 matches so we have at least one "definitely complete"
            # pre-token to emit (everything before the last match).
            if len(matches) <= 1:
                continue
            cut = matches[-1].start()     # start of the possibly-incomplete last match
            yield from self.encode(buf[:cut])
            buf = buf[cut:]               # keep the tail for next iteration

        # EOF: flush whatever remains. By now it's a complete final pre-token.
        if buf:
            yield from self.encode(buf)

    def decode(self, ids: list[int]) -> str:
        """Turn a list of token ids back into a string.

        We concatenate the raw bytes of every token, THEN decode the whole
        byte string as UTF-8 at the end. This matters because a multi-byte
        UTF-8 character (e.g. "é" = b'\\xc3\\xa9') can be split across two
        separate tokens — decoding each token individually would fail.

        errors='replace' substitutes U+FFFD for any genuinely invalid sequences
        instead of raising, which is the standard/expected BPE decoder behavior.
        """
        if not ids:
            return ""
        b = b"".join(self.vocab[i] for i in ids)
        return b.decode("utf-8", errors="replace")

    # ---------- Helper Functions ----------

    def _encode_plain(self, text: str) -> list[int]:
        """Encode a slice of text that contains NO special tokens.
        Runs pre-tok regex -> BPE -> id lookup for each pre-token.
        """
        ids: list[int] = []
        for matched in self.rx.finditer(text):
            chunk = matched.group()
            if not chunk:
                continue

            # BPE works on raw bytes, not unicode codepoints. Encode to UTF-8 first.
            chunk_bytes = chunk.encode("utf-8")
            for token in self._bpe(chunk_bytes):
                ids.append(self.byte_to_id[token])

        return ids

    def _bpe(self, pre_token_bytes: bytes) -> list[bytes]:
        """Apply BPE merges to a single pre-token.

        Algorithm (standard GPT-2 style):
          Start with the word as a list of single-byte tokens.
          Repeat:
            - Scan all adjacent pairs in the word.
            - Find the pair with the LOWEST merge rank (= earliest trained merge).
            - If no pair exists in merge_ranks, we're done.
            - Otherwise, merge every occurrence of that pair in one left-to-right pass.
          Return the final list of token bytes.

        Why "lowest rank globally" instead of "leftmost mergeable pair"?
        Because BPE was trained by applying merges in a specific order, and the
        encoder must reproduce that exact order to match training. Greedy
        leftmost can give a different tokenization on tricky inputs.
        """
        # Memoization: same pre-token -> same output, always.
        cached = self._bpe_cache.get(pre_token_bytes)
        if cached is not None:
            return cached

        # Initial state: each byte is its own token.
        # bytes([b]) gives a length-1 bytes object, not an int.
        word: list[bytes] = [bytes([b]) for b in pre_token_bytes]

        # Single-byte (or empty) pre-tokens have no pairs to merge.
        if len(word) <= 1:
            self._bpe_cache[pre_token_bytes] = word
            return word

        while True:
            best_pair, best_rank = None, None

            # Scan adjacent pairs to find the one with smallest merge rank.
            # `prev`/`cur` walk a sliding window of size 2 without rebuilding pairs.
            prev = word[0]
            for cur in word[1:]:
                p = (prev, cur)
                r = self.merge_ranks.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_pair = p
                prev = cur

            # No adjacent pair is in merges -> nothing left to merge, we're done.
            if best_pair is None:
                break

            a, b = best_pair
            new_token = a + b  # bytes concatenation

            # Second pass: rebuild `word`, collapsing every (a, b) adjacency into new_token.
            # We merge ALL occurrences of this pair in one go rather than one at a time,
            # which is correct because merging non-overlapping occurrences is independent.
            merged: list[bytes] = []
            i = 0
            L = len(word)
            while i < L:
                if i < L - 1 and word[i] == a and word[i + 1] == b:
                    merged.append(new_token)
                    i += 2  # skip both elements of the merged pair
                else:
                    merged.append(word[i])
                    i += 1
            word = merged

            # Optimization: once the word is a single token, no more merges possible.
            if len(word) <= 1:
                break

        self._bpe_cache[pre_token_bytes] = word
        return word

        
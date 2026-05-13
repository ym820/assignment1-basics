# BPE Tokenization — Deep Dive

A walkthrough of how Byte Pair Encoding (BPE) actually works at encode time,
with a fully worked example. Companion notes to `cs336_basics/tokenizer.py`.

---

## 1. What problem does BPE solve?

A language model needs a finite vocabulary, but natural text has effectively
infinite "words" (typos, names, rare compounds, code identifiers, other
languages, …). Two extremes:

- **Word-level tokenization**: small sequences, but huge vocab and tons of
  out-of-vocabulary (OOV) words.
- **Byte-level tokenization**: no OOV ever (every byte is known), but sequences
  become very long — a single English word is 4–8 tokens.

**BPE sits in between.** Start from raw bytes, then repeatedly merge the most
frequent adjacent pair into a new token. Common sequences (`" the"`, `"ing"`,
`" of"`) become single tokens; rare sequences fall back to finer-grained
bytes. No OOV possible, and sequences are ~3–4× shorter than pure bytes.

---

## 2. Two phases: training vs encoding

| Phase      | Input                    | Output                       | Runs when?        |
|------------|--------------------------|------------------------------|-------------------|
| Training   | A big text corpus        | `vocab` + ordered `merges`   | Once, offline     |
| Encoding   | A single string          | A list of token ids          | Every forward pass|

`tokenizer.py` only implements the **encoding** phase. Training lives in
`train_bpe.py`. This note focuses on encoding.

The two artifacts from training:

- **`vocab: dict[int, bytes]`** — id → token bytes. Contains every single byte
  0..255 plus every merged token created during training.
- **`merges: list[tuple[bytes, bytes]]`** — ordered list of merge rules.
  **Order matters**: index 0 was the first merge learned during training,
  index N was the last. Encoding must apply them in the same priority order.

---

## 3. Tiny worked example

Suppose we trained on text like `"low lower lowest"` and got:

### Vocab

```
0: b' '     1: b'l'    2: b'o'    3: b'w'
4: b'e'     5: b'r'    6: b'lo'   7: b'low'
8: b'er'    9: b'lower'
```

### Merges (priority order)

```
rank 0: (b'l', b'o')    → b'lo'
rank 1: (b'lo', b'w')   → b'low'
rank 2: (b'e', b'r')    → b'er'
rank 3: (b'low', b'er') → b'lower'
```

Think of `merges` as a recipe: "To tokenize, repeatedly find the
lowest-rank pair currently in the word and apply it."

### Encoding `"lower"` step by step

**Start** — split into single-byte tokens:
```
[b'l', b'o', b'w', b'e', b'r']
```

**Iteration 1** — scan all adjacent pairs, look each up in `merge_ranks`:

| pair      | rank |
|-----------|------|
| (l, o)    | 0    |
| (o, w)    | —    |
| (w, e)    | —    |
| (e, r)    | 2    |

Lowest rank is `(l, o)` at 0. Merge it:
```
[b'lo', b'w', b'e', b'r']
```

**Iteration 2** — new pairs:

| pair       | rank |
|------------|------|
| (lo, w)    | 1    |
| (w, e)     | —    |
| (e, r)     | 2    |

Lowest is `(lo, w)` at 1. Merge:
```
[b'low', b'e', b'r']
```

**Iteration 3** — pairs `(low, e)` (no rank) and `(e, r)` (rank 2). Merge `(e, r)`:
```
[b'low', b'er']
```

**Iteration 4** — pair `(low, er)` has rank 3. Merge:
```
[b'lower']
```

**Iteration 5** — only one token left, stop.

**Lookup ids:** `b'lower' → 9`.

**Result:** `encode("lower") == [9]`.

---

## 4. Why "lowest rank globally", not "leftmost"?

A tempting simpler algorithm: walk left-to-right, merge the first mergeable
pair you see, repeat. **This is wrong.** It can produce different output than
what training would imply, which means the model sees token sequences it was
never trained on.

### Counter-example sketch

Imagine merges:
```
rank 0: (b, c)
rank 1: (a, b)
```
and input `[a, b, c]`.

- **Leftmost**: sees `(a, b)` at rank 1, merges it → `[ab, c]`. No more merges possible. Output: `[ab, c]`.
- **Lowest-rank-globally (correct)**: sees `(a, b)` rank 1 *and* `(b, c)` rank 0. Rank 0 wins. Merge `(b, c)` → `[a, bc]`. No more merges. Output: `[a, bc]`.

Different tokenizations. Only the second matches what training order would
produce. This is why `_bpe` scans **all** pairs each iteration and picks the
minimum rank, even though it costs more.

---

## 5. Pre-tokenization: the regex before BPE

If you run BPE on a whole sentence like `"the cat sat"`, nothing stops it
from learning a merge like `(b'the', b' cat')`. That's bad — it balloons the
vocab with whole phrases and ruins generalization.

Solution: **split the string into "pre-tokens" first**, using a regex, and
apply BPE to each pre-token independently. Merges can never cross pre-token
boundaries because `_bpe` is called separately on each one.

The GPT-2 regex is:

```python
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

It matches, in order of preference:

1. `'s 'd 'm 't 'll 've 're` — English contractions stay attached to the preceding word.
2. ` ?\p{L}+` — optional leading space, then unicode letters. A "word" with its preceding space.
3. ` ?\p{N}+` — same for digits.
4. ` ?[^\s\p{L}\p{N}]+` — same for punctuation runs.
5. `\s+(?!\S)` — trailing whitespace (end of string / before newline).
6. `\s+` — any other whitespace run.

Example: `"Hello, world!"` → `["Hello", ",", " world", "!"]`. The leading
space on `" world"` is part of that pre-token by design — it means "space
followed by world" is a single unit, which is how GPT-2's vocab stores most
words.

---

## 6. Special tokens

Special tokens like `"<|endoftext|>"` must be treated as atomic — never
pre-tokenized, never merged. Two rules:

1. **Split on specials first**, before anything else. Everything between
   specials goes through the normal (regex + BPE) pipeline. Each special
   emits its id directly.
2. **Longest-first ordering.** If you have both `<|eot|>` and
   `<|eot|><|eot|>` as specials, the longer one must match first — otherwise
   the pair-of-eots would get tokenized as two single-eots. Regex alternation
   tries alternatives left-to-right, so sort by length descending before
   building the alternation pattern.

---

## 7. Decoding

Decoding is trivially easy compared to encoding:

```python
def decode(ids):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")
```

One subtlety: **concatenate the bytes first, THEN decode as UTF-8.** A single
multi-byte UTF-8 codepoint (e.g. `"é"` = `b'\xc3\xa9'`) might end up split
across two tokens during encoding. If you decoded each token individually,
you'd get UnicodeDecodeError on the half-characters. Concatenating first
reassembles them.

`errors="replace"` substitutes `U+FFFD` (`�`) for any genuinely invalid byte
sequences (e.g. truncated model output) instead of raising. This is the
expected default for BPE decoders.

---

## 8. Performance notes

The naive algorithm (scan all pairs → merge → repeat) is `O(n²)` per
pre-token in the worst case. That's fine here because:

1. **Pre-tokens are short.** Most are <10 bytes. `n²` of a small `n` is nothing.
2. **Caching dominates.** Natural text reuses the same pre-tokens over and
   over (`" the"` appears millions of times in a corpus). A single
   `dict[bytes, list[bytes]]` cache makes repeat calls free.

Faster algorithms exist (linked-list with a min-heap of pairs, achieving
`O(n log n)` per word), but they're only worth it for very long pre-tokens —
which the GPT-2 regex prevents from existing in the first place. So: cache
first, optimize only if profiling says so.

---

## 9. Streaming / `encode_iterable`

For very large files, you can't load everything into memory. The idea:

1. Read chunks (e.g. lines).
2. Append each chunk to a running buffer.
3. After each chunk, run the pre-tok regex on the buffer to find match boundaries.
4. Emit tokens for everything **up to the start of the last match** — the
   last match might still be incomplete (e.g. the chunk ended mid-word).
5. Keep the tail from the last match start onward in the buffer for the next iteration.
6. At EOF, flush whatever is left.

Why keep the last match? Consider the buffer containing `" hel"` when the
next chunk arrives with `"lo world"`. If you emitted `" hel"` immediately,
you'd tokenize it as separate bytes. Waiting until the next chunk turns it
into `" hello"` which BPE handles correctly.

This is subtle but load-bearing: get the cut point wrong and your streaming
encoder produces different ids than the non-streaming version on the same
input.

---

## 10. Mental model summary

When you read `tokenizer.py`, keep these anchors in mind:

- **Specials > pre-tok regex > BPE merges** is the priority order of splitting.
  Outer layers protect inner layers from touching things they shouldn't.
- **`merges` is an ordered recipe, not a set.** Rank = priority = training order.
- **BPE operates on bytes, not characters.** Unicode happens at the edges
  (`encode` input, `decode` output), never inside `_bpe`.
- **Cache everything.** The same pre-token will be encoded thousands of times.

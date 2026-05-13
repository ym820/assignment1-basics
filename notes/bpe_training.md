# BPE Training — Deep Dive

How to *learn* a BPE vocabulary from a corpus. Companion notes to
`cs336_basics/train_bpe.py`. Read `bpe_tokenization.md` first — this note
assumes you already understand how *encoding* uses `vocab` and `merges`.

---

## 1. The core algorithm (naive version)

BPE training is conceptually simple:

```
1. Start with every byte 0..255 as its own token.
2. Pre-tokenize the corpus into a {word: count} dict.
   Each word starts as a tuple of single-byte tokens.
3. Repeat until you have enough tokens:
     a. Count every adjacent pair across the whole corpus
        (weighted by word counts).
     b. Find the most frequent pair (A, B).
     c. Record the merge rule (A, B) → AB.
     d. Add AB to the vocabulary.
     e. Replace every occurrence of (A, B) with AB in all words.
```

That's the whole algorithm. Everything in `train_bpe.py` is engineering
around making steps 3a and 3e fast.

### Tiny worked example

Corpus (already pre-tokenized):
```
"low"    appears 5 times
"lower"  appears 2 times
"newest" appears 6 times
```

Initial state (each word as a tuple of single-byte tokens):
```
(l, o, w)          count 5
(l, o, w, e, r)    count 2
(n, e, w, e, s, t) count 6
```

**Iteration 1 — count pairs:**
```
(l,o) : 5 + 2         = 7
(o,w) : 5 + 2         = 7
(w,e) : 2 + 6         = 8
(e,r) : 2             = 2
(n,e) : 6             = 6
(e,s) : 6             = 6
(s,t) : 6             = 6
```
Most frequent: `(w, e)` with count 8. Merge.

New state:
```
(l, o, w)       count 5
(l, o, w, e, r) count 2  ← wait, (w,e) is here!
```
Hmm, I was sloppy — `(l,o,w,e,r)` does contain `(w,e)`. Let me redo:
```
(l, o, w)          count 5
(l, o, we, r)      count 2   ← (w,e) merged
(n, e, we, s, t)   count 6   ← (w,e) merged
```

**Iteration 2 — count pairs again:**
```
(l, o): 5 + 2 = 7       ← unchanged
(o, w): 5     = 5       ← decreased! was 7, now only in "low"
(o, we): 2    = 2       ← new pair
(we, r): 2    = 2       ← new pair
(n, e): 6
(e, we): 6              ← new pair
(we, s): 6              ← new pair
(s, t): 6
```
Most frequent: `(l, o)` at 7. Merge.

And so on. You can see that each merge mostly affects only a few pairs
near the merge site — the pairs `(n,e)` and `(s,t)` never changed between
iterations 1 and 2. **This observation is what makes the efficient
implementation possible.**

---

## 2. Why the naive version is too slow

Suppose your corpus has:
- $W$ unique words (~millions for real corpora)
- $V$ target vocab size (~32k for GPT-2, ~100k+ for modern models)
- average word length $L$ (~5 bytes)

The naive version re-counts every pair from scratch each iteration:

$$
\text{Time} = V \times W \times L \approx 32{,}000 \times 1{,}000{,}000 \times 5 = 1.6 \times 10^{11}
$$

At ~10 million simple operations per second in pure Python, that's **hours
to days** per training run. Unacceptable.

The efficient version brings this down to roughly $V \times (\text{affected
pairs per merge})$, which is typically 3–4 orders of magnitude faster on
real corpora.

---

## 3. The efficient design: incremental updates

The key realization: **a merge only changes pair counts near the merge site.**
If we merge `(A, B)` → `AB` inside word `(X, A, B, Y)`, the pairs that
change are:

- `(X, A)` disappears, `(X, AB)` appears
- `(A, B)` disappears entirely (that's the merge itself)
- `(B, Y)` disappears, `(AB, Y)` appears

Every other pair in the corpus is untouched. So instead of recomputing
everything, we maintain two indices and update them incrementally.

### Data structures

```python
pair_counts: dict[pair, int]       # pair → total count across corpus
pair_from:   dict[pair, set[wid]]  # pair → set of word IDs containing it
word_id_to_word: dict[wid, tuple]  # stable word ID → current token tuple
word_counts: dict[wid, int]        # word ID → how often it appears
heap: max-heap of (count, pair)    # fast "most frequent pair" lookup
```

### Why stable word IDs?

You could use the word tuple itself as the key everywhere. But every merge
changes the tuple, which means every `pair_from` set that referenced it
would need updating — turning an O(1) lookup into O(pairs per word).

By assigning each unique pre-token a stable integer ID at the start and
never changing it, `pair_from` entries for unchanged pairs don't need any
maintenance during a merge — they still point at the same ID, which now
resolves to the updated word via `word_id_to_word[wid]`. This is the
single most important optimization in the whole file.

### Why `pair_from`?

Without it, when we merge `(A, B)`, we'd have to scan every word in the
corpus to find which ones contain the pair. With `pair_from[(A, B)]` we
go directly to the (usually small) set of affected words.

### Why a heap?

Every merge iteration we need the max-count pair. With ~100k unique pairs
and ~30k merges, a linear scan each iteration is 3 billion comparisons.
A heap makes it $O(\log n)$ per operation.

---

## 4. Lazy deletion — why the heap can grow

Here's the trick: when `pair_counts[(A, B)]` changes from 10 to 7, we
**don't** find and remove the old entry from the heap (that would be
O(n)). Instead we push a *new* entry `(-7, (A, B))` and leave the stale
`(-10, (A, B))` in the heap.

When we later pop `(-10, (A, B))`, we check:
```python
pair_counts[(A, B)] == 7  ≠  -(-10) == 10
```
So this entry is **stale**. Discard it, pop again. Eventually we hit
`(-7, (A, B))` which matches, and we return it.

**Invariant:** the top of the heap either matches `pair_counts` (fresh) or
doesn't (stale). Stale entries are harmless — they get skipped over. Each
entry is popped at most once, so total work is bounded even though the
heap gets bigger than `len(pair_counts)`.

---

## 5. Tie-breaking: `_ReverseKey`

Two pairs may have the same count. Which one wins?

The reference implementation (and the assignment tests) breaks ties toward
the **lexicographically largest** bytes tuple. E.g. if `(b'a', b'b')` and
`(b'x', b'y')` both have count 5, we pick `(b'x', b'y')`.

But Python's min-heap, when two entries have the same `-count`, compares
the *next* element of the tuple. Default bytes comparison picks the
**smallest**, giving us `(b'a', b'b')` — wrong.

`_ReverseKey` wraps the pair with inverted `<` and `<=`, so "smallest in
heap order" now means "lexicographically largest in bytes order." Yes,
it's a little cursed. It's also exactly one short class.

---

## 6. The adjacent-merge special case

Consider the word `(a, b, a, b)` merging pair `(a, b)`. Both occurrences
merge at once, producing `(ab, ab)`. What happens to the pair `(b, a)`
that was between them?

In the general case, when we merge at position `p`, the RIGHT neighbor is
`(word[p+1], word[p+2])`, which becomes `(new_token, word[p+2])`. But if
`p+2` is ALSO a merge position, then `word[p+2] = token1` and
`word[p+3] = token2`, and both are being consumed. The seam pair is
`(token2, token1)` (the `b` at p+1 and the `a` at p+2), and it becomes
`(new_token, new_token)`.

And at the second merge position `p+2`, its LEFT neighbor is the same
seam — we must NOT process it again, or we'd double-count the shift.

```python
# Left neighbor (skip if the previous position is also a merge site)
if p > 0 and (p - 2) not in pos_set:
    ...

# Right neighbor
if p + 2 < len(old_word):
    if (p + 2) in pos_set:
        # Adjacent merge: seam becomes (new_token, new_token)
        shift((token2, token1), (new_token, new_token), ...)
    else:
        shift((token2, old_word[p+2]), (new_token, old_word[p+2]), ...)
```

This edge case is easy to miss and not well-exercised by small test cases.
It matters for any corpus that contains repetition like `"hahaha"` or
`"lalalala"`.

---

## 7. The `pair_from` set-difference trick

Why do we update `pair_from` using set difference at the end of
`_merge_in_word`, instead of updating it inside the neighbor loop?

Consider `old_word = (a, b, c, a, b)` merging pair `(b, c)` at position 1.

In the neighbor loop, the LEFT neighbor is `(a, b)` — the pair that used
to sit at position `(0, 1)`. You might think: "this pair disappeared, so
discard `wid` from `pair_from[(a, b)]`."

**Wrong!** The word also contains `(a, b)` at positions `(3, 4)`, which
survives the merge untouched. `new_word = (a, bc, a, b)` still contains
`(a, b)`. Discarding `wid` would break the invariant.

Set difference sidesteps the entire problem:
```python
old_pair_set = set(zip(old_word, old_word[1:]))
new_pair_set = set(zip(new_word, new_word[1:]))
for pair in old_pair_set - new_pair_set:   # truly disappeared
    pair_from[pair].discard(wid)
for pair in new_pair_set - old_pair_set:   # truly appeared
    pair_from[pair].add(wid)
```

We only touch `pair_from` for pairs whose *membership* changed, not for
pairs whose *count* changed. These are different sets of pairs, and
conflating them causes subtle bugs.

---

## 8. Pre-tokenization and special tokens

Before BPE even starts, we split the corpus into "pre-tokens" using the
GPT-2 regex (see `bpe_tokenization.md` section 5). This matters because:

- Merges can never cross pre-token boundaries. No `(b'the', b' cat')`.
- It's trivially parallelizable — chunk the input file, run each chunk
  independently, merge the counts at the end.
- Pre-token frequencies are what actually drives which merges get learned.

**Special tokens** (e.g. `<|endoftext|>`) are split OUT of the text before
anything else, using `re.split(st_pattern, chunk)`. This drops them from
the training text entirely — they get added directly to the vocab as
reserved IDs (e.g. `vocab[256] = b"<|endoftext|>"`) and never appear in
any merge.

---

## 9. Parallelization (the easy part)

Pre-tokenization is embarrassingly parallel. Split the file into byte
ranges (being careful not to split in the middle of a `<|endoftext|>`
marker), hand each range to a worker process, each worker returns a
`{pre_token: count}` dict, the main process merges them.

The **merge loop** is NOT parallelized in our implementation — the indices
(`pair_counts`, `pair_from`, `heap`) are shared mutable state, and the
coordination overhead would eat the gains. Real production trainers
(HuggingFace `tokenizers` in Rust) do parallelize across words within a
merge step, but that's out of scope here.

A detail in the parallel path: workers return `dict[bytes, int]`, not
`dict[tuple[bytes, ...], int]`. Why? Raw `bytes` is ~40% cheaper to pickle
than a tuple of single-byte objects. Conversion happens in the reducer
after the workers return.

---

## 10. Complexity summary

Let $V$ = target vocab size, $W$ = unique pre-tokens, $L$ = avg word length,
$k$ = avg affected words per merge (small in practice — most merges only
touch a handful of high-frequency words).

| Phase                       | Naive          | Efficient           |
|-----------------------------|----------------|---------------------|
| Initial pair counting       | $O(WL)$        | $O(WL)$             |
| Find best pair              | $O(V \cdot P)$ | $O(V \log P)$       |
| Apply one merge             | $O(WL)$        | $O(k \cdot \text{neighbors})$ |
| **Total**                   | $O(V W L)$     | $\approx O(WL + V(k + \log P))$ |

where $P$ = unique pairs at a given time. On a real corpus this is
thousands of times faster.

---

## 11. Mental model summary

- **BPE training is "repeatedly merge the most common pair."** Everything
  else is bookkeeping.
- **Stable word IDs** let pair_from entries for unchanged pairs survive
  merges without any maintenance.
- **Lazy deletion** lets the heap handle frequent count updates without
  O(n) edits.
- **Incremental updates** touch only the few pairs near each merge site,
  not the whole corpus.
- **Set difference for membership** correctly handles the case where a
  pair appears in a word at multiple positions, only some of which are
  consumed by the merge.
- **Special tokens never participate in training** — they're sliced out
  before pre-tokenization and added to the vocab as reserved IDs.

When you look at `train_bpe.py` again, each function maps to one of these
ideas:

| Function             | Job                                           |
|----------------------|-----------------------------------------------|
| `_count_pretokens`   | Step 2: turn text into {word: count}          |
| `_push_pair` / `_pop_best_pair` | Heap operations with lazy deletion |
| `_shift_pair`        | The atomic "decrement old, increment new" op  |
| `_merge_in_word`     | Apply one merge to one word + update indices  |
| `_apply_merge`       | Fan out `_merge_in_word` across affected words|
| `train_bpe`          | Orchestration                                 |

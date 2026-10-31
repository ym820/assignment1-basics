## Problem (unicode1): Understanding Unicode (1 point)

**(a)** What Unicode character does `chr(0)` return?
**Answer:** 
`chr(0)` returns `\x00`. The NULL character, Unicode code point U+0000

**(b)** How does this character’s string representation `(__repr__())` differ from its printed representation?
**Answer:**
- **`repr(chr(0))`** → `"'\\x00'"` — shows the escaped form `\x00` so it's visible/unambiguous.
- **`print(chr(0))`** → outputs the raw NULL byte itself — usually invisible, no visible glyph on screen (though some terminals show nothing or a placeholder).

**(c)** What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations:
```bash
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```
**Answer:**
```bash
>>> chr(0)
'\x00'
>>> print(chr(0))

>>> "this is a test" + chr(0) + "string"
'this is a test\x00string'
>>> print("this is a test" + chr(0) + "string")
this is a teststring
```
Python treats `\x00` as just another character in the string — it doesn't terminate the string (unlike C, where NULL ends a string). But when **displayed**, it's invisible, which can be misleading/confusing since the string is actually longer than it looks.

---
## Problem (unicode2): Unicode Encodings (3 points)

**(a)** What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.
**Answer:**
- **No wasted/null bytes** — ASCII chars in UTF-8 take 1 byte; in UTF-16 they take 2 bytes (with a null byte `\x00`), in UTF-32 they take 4 bytes (3 nulls). E.g. "A" → UTF-8: `41`, UTF-16: `41 00`, UTF-32: `41 00 00 00`. Those null bytes are wasted vocabulary slots and wasted sequence length — bad for BPE, which learns from byte co-occurrence statistics.
- **Smaller, denser alphabet (256 possible byte values)** — UTF-8's base vocabulary is exactly 256 bytes, a clean, compact starting point for BPE merges. UTF-16/32 introduce more distinct byte values and irregular patterns (depending on endianness), making merge statistics noisier and less consistent.
- **Variable-length efficiency** — UTF-8 uses 1 byte for common ASCII (English text, code, punctuation) and only expands to 2–4 bytes for rarer/non-Latin characters. This keeps common-case sequences short. UTF-16/32 don't give you this — you pay the larger fixed/near-fixed cost even for simple text.
- **Backward compatibility with ASCII** — since most training data (code, English) is ASCII-heavy, UTF-8 byte sequences look nearly identical to plain ASCII text, so BPE merges learned are more meaningful/interpretable and generalize better.
- **No endianness issues** — UTF-16/32 require choosing BE or LE, adding inconsistency across sources; UTF-8 has no such ambiguity.

**(b)** Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.
```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
	return "".join([bytes([b]).decode("utf-8") for b in bytestring])

>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```
**Answer:**
It decodes each byte separately, but multi-byte UTF-8 characters can't be decoded one byte at a time. Individual bytes of a multi-byte sequence aren't valid UTF-8 on their own. 
Example
```python
decode_utf8_bytes_to_str_wrong("café".encode("utf-8"))
# UnicodeDecodeError: 'é' is encoded as bytes b'\xc3\xa9',
# and decoding \xc3 or \xa9 alone is invalid UTF-8
```

**(c)** Give a two byte sequence that does not decode to any Unicode character(s).
**Answer:**
`b'\x80\x80'` — these are two continuation bytes with no leading byte, which is invalid UTF-8 and raises a `UnicodeDecodeError`.

---
## Problem (`train_bpe_tinystories`): BPE Training on TinyStories
**(a)** Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary size of 10,000. Make sure to add the TinyStories <|endoftext|> special token to the vocabulary. Serialize the resulting vocabulary and merges to disk for further inspection. How much time and memory did training take? What is the longest token in the vocabulary? Does it make sense?
**Answer:**
 Tested two implementations: 

| Impl                             | Time    | Peak memory | Merges | Longest token        |
| -------------------------------- | ------- | ----------- | ------ | -------------------- |
| [[train_bpe.py]] (heap)          | 31.7 s  | 118 MB      | 9743   | `b' accomplishment'` |
| [[train_bpe_parallel.py]] (scan) | 124.4 s | 55 MB       | 9743   | `b' accomplishment'` |
heap version is faster than scan version while the trade-off is the memory. The 

**(b)** Profile your code. What part of the tokenizer training process takes the most time?
**Answer:**

Breakdown of two versions:

| Phase             | Heap        | Scan        |
| ----------------- | ----------- | ----------- |
| Pretokenization   | 20.7s (61%) | 48.6s (33%) |
| Merge selection   | 1.6s        | 87.1s (60%) |
| Merge application | 10.6s       | 8s          |
| Total             | 33.9s       | 146.4s      |
The `max()` function in Scan version is the bottleneck. It's reduced in Heap version.

---
## Problem (`train_bpe_expts_owt`): BPE Training on OpenWebText
**(a)** Train a byte-level BPE tokenizer on the OpenWebText dataset, using a maximum vocabulary size of 32,000. Serialize the resulting vocabulary and merges to disk for further inspection. What is the longest token in the vocabulary? Does it make sense?
**Answer:**

| Impl             | Time     | Peak memory | Merges | Longest token                      |
| ---------------- | -------- | ----------- | ------ | ---------------------------------- |
| [[train_bpe.py]] | 31.8 min | 13.8 GB     | 31743  | `ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ` |
The longest token is likely due to the noisy data from the web

**(b)** Compare and contrast the tokenizer that you get training on TinyStories versus OpenWebText.
**Answer:**
**TinyStories tokenizer:**
- Small, simple vocabulary — mostly common English words, short/simple subwords
- Fewer merges needed since text uses simple vocabulary and repetitive sentence patterns
- Higher compression ratio for its own domain, but merges reflect narrow, simple vocabulary (no code, rare words, technical terms, multilingual text)
- Trained vocab converges faster to whole-word tokens due to limited word diversity

**OpenWebText tokenizer:**
- Much more diverse vocabulary — web text includes code, URLs, punctuation patterns, numbers, technical jargon, multiple languages/scripts
- Needs more merges to cover this diversity, so vocabulary spreads across many more distinct token types
- Individual tokens tend to be shorter/more fragmented on average (more subword splitting) since rare/varied words don't merge into single tokens as easily
- More robust/generalizable — but less efficient (lower compression) specifically on simple children's-story-style text

---
## Problem (`tokenizer_experiments`): Experiments with tokenizers

**(a)** Sample 10 documents from TinyStories and OpenWebText. Using previously-trained TinyStories and OpenWebText tokenizers (10K and 32K vocabulary size, respectively), encode these sampled documents into integer IDs. What is each tokenizer’s compression ratio (bytes/token)?
**Answer:**
Tinystories: 4.15 bytes/token
OWT: 4.43 bytes/token

**(b)** What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer? Compare the compression ratio and/or qualitatively describe what happens.
**Answer:**
Tokenize OWT sample with Tinystories tokenizer: 3.24 bytes/token
Tokenize Tinystories sample with OWT tokenizer: 3.98 bytes/token
Both compression ratio drops when tokenizing on dataset that it is not trained on

**(c)** Estimate the throughput of your tokenizer (e.g., in bytes/second). How long would it take to tokenize the Pile dataset (825GB of text)?
**Answer:**
Tinystories: 1590406.23 bytes/second
OWT: 1313909.63 bytes/second
Estimated time to tokenize 825 GB text:
Tinystories: 6.00 days
OWT: 7.27 days

---
## Problem (transformer_accounting): Transformer LM resource accounting

**(a)** Consider a GPT-2 XL-sized model using our assignment architecture, which has the following configuration:
	**vocab_size**: 50,257
	**context_length**: 1,024
	**num_layers**: 48
	**d_model**: 1,600
	**num_heads**: 25
	**d_ff**: 4,288 (the nearest multiple of 64 to 8/3 × 1, 600)
	Suppose we constructed our model using this configuration. How many trainable parameters would our model have? Assuming each parameter is represented using single-precision floating point, how much memory is required to just load this model?

**Answer**

Let
- B: `batch_size`
- V: `vocab_size`
- S: `context_length`
- L: `num_layers`
- d: `d_model`
- h: `num_heads`
- d_ff: $\frac{8}{3}d$

| Component                         | Count                                             |
| --------------------------------- | ------------------------------------------------- |
| Token embedding                   | $Vd$                                              |
| Per layer: $W_Q, W_K, W_V, W_O$   | $4d^2$                                            |
| Per layer: SwiGLU $W_1, W_2, W_3$ | $3 \cdot d \cdot d_{ff}$                          |
| Per layer: two RMSNorm gains      | $2d$                                              |
| Final RMSNorm                     | $d$                                               |
| Output embedding (LM head)        | $Vd$                                              |
| **Total** $P$                     | $2Vd + L(4d^2 + 2d + 3 \cdot d \cdot d_{ff}) + d$ |

**Total** = 1640452800 ~= **1.64B**

1640452800 parameters * 4 bytes ~= **6.56 GB**

**(b)** Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped model. How many FLOPs do these matrix multiplies require in total? Assume that our input sequence has context_length tokens.

**Answer**

| Component                                        | FLOPs                                 |
| ------------------------------------------------ | ------------------------------------- |
| Q, K, V projections                              | $3 \cdot 2BSd^2 = 6BSd^2$             |
| $QK^\top$ scores, summed over heads ($d = hd_k$) | $2BS^2d$                              |
| Attention weights times $V$                      | $2BS^2d$                              |
| Output projection                                | $2BSd^2$                              |
| SwiGLU $W_1, W_2, W_3$                           | $3 \cdot 2BSd\,d_{ff} = 6BSd\,d_{ff}$ |
| **Per-layer total**                              | $8BSd^2 + 4BS^2d + 6BSd\,d_{ff}$      |

Last layer projection to $V$
- (S, d) · (d, vocab_size) -> (S, vocab_size)
- $F_{\text{lm}} = BSdV$

$$F_{\text{fwd}} = L\left(8BSd^2 + 4BS^2d + 6BSd\,d_{ff}\right) + 2BSdV$$When B=1, $F_{\text{fwd}} = 3516769894400 = 3.52 \text{ TFLOPs}$


**(c)** Based on your analysis above, which parts of the model require the most FLOPs?

**Answer**
FFN takes most FLOPs

**(d)** Repeat your analysis with GPT-2 small (12 layers, 768 d_model, 12 heads), GPT-2 medium (24 layers, 1024 d_model, 16 heads), and GPT-2 large (36 layers, 1280 d_model, 20 heads). As the model size increases, which parts of the Transformer LM take up proportionally more or less of the total FLOPs?

**Answer**
GPT-2 small (12 layers, 768 d_model, 12 heads):
d_ff = 8/3 * 768 to nearest multiple of 64 = 2048

|             part             |      FLOPs      |    %    |
|------------------------------|----------------:|--------:|
| Attention projections (8Sd²) |  57,982,058,496 |  19.88% |
| Attention matmul (4S²d)      |  38,654,705,664 |  13.25% |
| FFN (6S·d·d_ff)              | 115,964,116,992 |  39.76% |
| LM head (2S·d·V), once       |  79,047,426,048 |  27.10% |
| Total                        | 291,648,307,200 | 100.00% |

GPT-2 medium (24 layers, 1024 d_model, 16 heads):
d_ff = 8/3 * 1024 to nearest multiple of 64 = 2752

|             part             |      FLOPs      |    %    |
|------------------------------|----------------:|--------:|
| Attention projections (8Sd²) | 206,158,430,208 |  24.83% |
| Attention matmul (4S²d)      | 103,079,215,104 |  12.42% |
| FFN (6S·d·d_ff)              | 415,538,085,888 |  50.05% |
| LM head (2S·d·V), once       | 105,396,568,064 |  12.70% |
| Total                        | 830,172,299,264 | 100.00% |

GPT-2 large (36 layers, 1280 d_model, 20 heads):
d_ff = 8/3 * 1280 to nearest multiple of 64 = 3392

|             part             |       FLOPs       |    %    |
|------------------------------|------------------:|--------:|
| Attention projections (8Sd²) |   483,183,820,800 |  27.32% |
| Attention matmul (4S²d)      |   193,273,528,320 |  10.93% |
| FFN (6S·d·d_ff)              |   960,327,843,840 |  54.30% |
| LM head (2S·d·V), once       |   131,745,710,080 |   7.45% |
| Total                        | 1,768,530,903,040 | 100.00% |

Assuming same sequence length, 
- Per layer, FFN FLOPs scales with d_model * d_ff, which is about 8/3 * d_model^2
- Per layer, Attention projections scales with d_model^2
- Per layer, Attention matmul scales with d_model
- LM head scale with d_model and vocab size

As model size increases (both width and depth), FFN and attention projection take up a larger share of total FLOPs; Attention matmul decreases; LM head decreases drastically

**(e)** Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for one forward pass change? How does the relative contribution of FLOPs of the model components change?

**Answer**
The total FLOPs increases from 3.5 TFLOPs to 133.6 TFLOPs, which is around 38x for a 16x context length. The linear terms (projection, FNN, and LM head) grow 16×, while the S² term (attention matmul) grows 256×.

| part                         | context 1,024 | context 16,384 |
|------------------------------|--------------:|---------------:|
| Attention projections (8Sd²) |        28.62% |         12.06% |
| Attention matmul (4S²d)      |         9.16% |         61.73% |
| FFN (6S·d·d_ff)              |        57.53% |         24.24% |
| LM head (2S·d·V), once       |         4.68% |          1.97% |
| Total FLOPs                  |      3.52 TFLOPs |    133.6 TFLOPs |

---

## Problem (`learning_rate_tuning`): Tuning the learning rate

As we will see, one of the hyperparameters that affects training the most is the learning rate. Let’s see that in practice in our toy example. Run the SGD example above with three other values for the learning rate: 1e1, 1e2, and 1e3, for just 10 training iterations. What happens with the loss for each of these learning rates? Does it decay faster, slower, or does it diverge (i.e., increase over the course of training)?

**Answer:**
When `lr=10`, the loss decays slower; when `lr=100`, the loss decays faster towards 0; when `lr=1000`, the loss diverges.
```
Running SGD on learning rate: 10.0
step 00: loss = 26.271406
step 01: loss = 16.813700
step 02: loss = 12.394342
step 03: loss = 9.697249
step 04: loss = 7.854772
step 05: loss = 6.512506
step 06: loss = 5.492435
step 07: loss = 4.693442
step 08: loss = 4.053156
step 09: loss = 3.530750
Running SGD on learning rate: 100.0
step 00: loss = 21.879858
step 01: loss = 21.879854
step 02: loss = 3.753989
step 03: loss = 0.089841
step 04: loss = 0.000000
step 05: loss = 0.000000
step 06: loss = 0.000000
step 07: loss = 0.000000
step 08: loss = 0.000000
step 09: loss = 0.000000
Running SGD on learning rate: 1000.0
step 00: loss = 32.328602
step 01: loss = 11670.625000
step 02: loss = 2015700.500000
step 03: loss = 224225200.000000
step 04: loss = 18162237440.000000
step 05: loss = 1146245218304.000000
step 06: loss = 58844487090176.000000
step 07: loss = 2531741755506688.000000
step 08: loss = 93314598366609408.000000
step 09: loss = 2996435267503521792.000000
```

---

## Problem (`adamw_accounting`): Resource accounting for training with AdamW

Let us compute how much memory and compute running AdamW requires. Assume we are using float32 for every tensor.

**(a)** How much peak memory does running AdamW require? Decompose your answer based on the memory usage of the parameters, activations, gradients, and optimizer state. Express your answer in terms of the batch_size and the model hyperparameters (vocab_size, context_length, num_layers, d_model, num_heads). Assume $d_{ff} =8/3 × d_{model}$.

**Answer**
Let
- B: `batch_size`
- V: `vocab_size`
- S: `context_length`
- L: `num_layers`
- d: `d_model`
- h: `num_heads`
- d_ff: $\frac{8}{3}d$

The number of parameters for a transformer is:
$$
\begin{aligned}
P &= V \cdot d + L \cdot (4 \cdot d \cdot d + 2 \cdot d + (3 \cdot d_{ff} \cdot d)) + d + d \cdot V \\
 &= L(12d^2 + 2d) + d + 2Vd
\end{aligned}
$$
**Parameters:**
The memory for the parameters is
$$
M_{params} = 4P \text{ bytes}
$$
**Gradients:**
There is one gradient per parameter:
$$
M_{grads} = 4P \text{ bytes}
$$
**Optimizer state:**
AdamW keeps two moments, `m` and `v`, for every parameter:
$$
M_{opt} = 2 \cdot 4P = 8P \text{ bytes}
$$
**Activations:**

**Per Transformer block,**

| Component                                                     | Elements stored                       |
| ------------------------------------------------------------- | ------------------------------------- |
| 2 RMSNorm outputs                                             | $2BSd$                                |
| Q, K, V projections                                           | $3BSd$                                |
| $QK^\top$ scores                                              | $BhS^2$                               |
| Softmax probabilities                                         | $BhS^2$                               |
| Weighted sum of values                                        | $BSd$                                 |
| Output projection                                             | $BSd$                                 |
| $W_1$ output, $W_3$ output, SiLU output, element-wise product | $4 \cdot BSd_{ff} = \tfrac{32}{3}BSd$ |
| $W_2$ output                                                  | $BSd$                                 |

Per-layer total:

$$\tfrac{56}{3}BSd + 2BhS^2$$

**After the blocks:**

- Final RMSNorm: $BSd$
- Logits from the output embedding: $BSV$
- Cross-entropy softmax/log-probs over the logits: $BSV$ (ignoring per-token loss $BS$ becuase $V$ is large)

$$A = L\left(\tfrac{56}{3}BSd + 2BhS^2\right) + BSd + 2BSV$$

$$M_{\text{act}} = 4A = L\left(\tfrac{224}{3}BSd + 8BhS^2\right) + 4BSd + 8BSV \text{ bytes}$$

**Total peak memory:**
$$
\begin{aligned}
M_{total} &= 4P + 4P + 8P + 4A \\
 &= 16P + 4A \\
 &= 16[L(12d^2 + 2d) + d + 2Vd] + 4[L\left(\tfrac{56}{3}BSd + 2BhS^2\right) + BSd + 2BSV]
\end{aligned}
$$

**(b)** Instantiate your answer for a GPT-2 XL-shaped model to get an expression that only depends on the batch_size. What is the maximum batch size you can use and still fit within 80GB memory?

For a GPT-2 XL, where `V=50,257`, `S=1,024`, `L=48`, `d=1600`, `h=25`

$$
\begin{aligned}
M_{total} &= 16P + 4A \\
 &= 16[L(12d^2 + 2d) + d + 2Vd] + 4[L\left(\tfrac{56}{3}BSd + 2BhS^2\right) + BSd + 2BSV] \\
 &= 26,168,601,600 + 16,356,614,144 \cdot B \\
\end{aligned}
$$

$$ B_{max} = \lfloor \tfrac{80 \times 10^9 - 26,168,601,60}{16,356,614,144} \rfloor = 3 $$
**(c)** How many FLOPs does running one step of AdamW take?

**Answer**

A `step` in AdamW involves:

| #   | Step              | Expression                                                          | Ops on one element                     | FLOPs  |
| --- | ----------------- | ------------------------------------------------------------------- | -------------------------------------- | ------ |
| 0   | Gradient          | $g \leftarrow \nabla_\theta \ell(\theta)$                           | (from backward)                        | -      |
| 1   | First moment      | $m \leftarrow \beta_1 m + (1-\beta_1) g$                            | 2 mul + 1 add                          | $3P+1$ |
| 2   | Second moment     | $v \leftarrow \beta_2 v + (1-\beta_2) g^2$                          | 3 mul + 1 add                          | $4P+1$ |
| 3   | Bias-corrected LR | $\alpha_t \leftarrow \alpha \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}$ | (contant ops)                          | $7$*   |
| 4   | Adam update       | $\theta \leftarrow \theta - \alpha_t \frac{m}{\sqrt{v}+\epsilon}$   | 1 sqrt + 1 add + 1 div + 1 mul + 1 sub | $5P$   |
| 5   | Weight decay      | $\theta \leftarrow \theta - \alpha\lambda\theta$                    | 1 mul + 1 sub                          | $2P$   |
|     | **Total**         |                                                                     |                                        | $14P$  |
<small> *we can save betas at $t-1$ so we don't need to recalculate every step </small>

Substituting $P=1.64B$, we get about 23 GFLOPs. 
We assume $F_{\text{bwd}} \approx 2 F_{\text{fwd}}$, so a forward and backward pass take about $3 \cdot 3.5 = 10.5$ TFLOPs 
Then one optimizer's compute takes about $2 \times 10^{-6}$ of a training step

**(d)** Model FLOPs utilization (MFU) is defined as the ratio of observed throughput (tokens per second) relative to the hardware’s theoretical peak FLOP throughput \[A. Chowdhery et al., 2022\]. An NVIDIA H100 GPU has a theoretical peak of 495 teraFLOP/s for “float32” (actually TensorFloat-32, which in reality is “bfloat19”) operations. Assuming you are able to get 50% MFU, how long would it take to train a GPT-2 XL for 400K steps and a batch size of 1024 on a single H100? Following J. Kaplan et al. and J. Hoffmann et al., assume that the backward pass has twice the FLOPs of the forward pass.

**Answer:**

When B = 1024, step s = 400k,
$$
\begin{aligned}
F_{\text{total}} &= 400,000 \cdot 1024 \cdot (F_{\text{fwd}} + F_{\text{bwd}}) \\
&= 400,000 \cdot 1024 \cdot (F_{\text{fwd}} + 2F_{\text{fwd}})\\
&= 400,000 \cdot 1024 \cdot 3 \cdot 3516769894400 \\
&= 4321406846238720000000 \\
&\approx 4.32 \times 10^{21}
\end{aligned}
$$

On a single H100, assuming 50% MFU

$$
T = \tfrac{T_{\text{total}}}{0.5 * 495 * 10^{12}} \approx 17460229.68 \text{ seconds} \approx 202 \text{ days}
$$
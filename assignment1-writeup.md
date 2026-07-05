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
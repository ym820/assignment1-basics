## Problem (unicode1): Understanding Unicode (1 point)
#### (a) What Unicode character does `chr(0)` return?
**Answer:**
`\x00`
#### (b) How does this character’s string representation `(__repr__())` differ from its printed representation?
**Answer:**
The character's string representation is the Unicode string `\x00` while the printed representation is rendered to be a space
#### (c) What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations:
```bash
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```
**Answer:**
When printed, the character is converted to a space while it remains as `\x00` in the raw string representation

## Problem (unicode2): Unicode Encodings (3 points)

#### (a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.
**Answer:**
First of all, UTF-8 is the dominant encoding of the Internet. It is convenient and cheap to use it directly without converting to UTF-16 or UTF-32. Second, the vocabulary size of UTF-8 is smallest, making it more manageable.

#### (b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.
```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
	return "".join([bytes([b]).decode("utf-8") for b in bytestring])

>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```
**Answer:**
This function will be wrong if the Unicode string is encoded with more than one UTF-8 byte string. For example, input `你好` will throw an error:
```python
>>> decode_utf8_bytes_to_str_wrong("你好".encode("utf-8"))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 2, in decode_utf8_bytes_to_str_wrong
  File "<stdin>", line 2, in <listcomp>
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe4 in position 0: unexpected end of data
```
This is because the character `"你"` has three UTF-8 bytes: `b'\xe4\xbd\xa0'` and looping through the byte string will try to decode the first byte `b'\xe4` which can't be decoded to a Unicode string.

#### (c) Give a two byte sequence that does not decode to any Unicode character(s).
**Answer:**
`\xe4\xbd` because it is the first two bytes of the 3-byte sequence for character `你`

from typing import Iterable, Iterator
import regex as re
import pickle

GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    """Byte-level BPE tokenizer, compatible with GPT-2 style vocab + merges."""

    def __init__(self,
                 vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None
                 ):
        self.vocab: dict[int, bytes] = vocab
        self.merges: list[tuple[bytes, bytes]] = merges
        self.special_tokens = special_tokens or []

        if self.special_tokens:
            # Say ["<|endoftext|>", "<|endoftext|><|endoftext|>"], 
            # if the short one comes first, the double token never matches as a unit.
            # So we want to sort special tokens from longest to shortest
            self.special_tokens: list[str] = sorted(special_tokens, key=len, reverse=True)
            self.st_pattern = "|".join(re.escape(tok) for tok in self.special_tokens)
        else:
            self.st_pattern = None

        self.merge_rank = {
            pair: idx for idx, pair in enumerate(self.merges)
        }
        self.byte_to_id: dict[bytes, int] = {
            b: i for i, b in self.vocab.items()
        }

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
        """Encode a full string into a list of token ids."""
        if not text:
            return []
        
        token_ids: list[int] = []
        # --- Pre-tokenization ---
        # The f'({pattern})' makes sure the special tokens are not gone
        parts = re.split(f'({self.st_pattern})', text) if self.st_pattern else [text]
        for part in parts:
            if part in self.special_tokens:
                token_ids.append(self.byte_to_id[part.encode('utf-8')])
            else:
                for pre_token in re.finditer(GPT2_PATTERN, part):
                    word_bytes = pre_token.group().encode("utf-8")
                    # tokens = self._bpe_1(word_bytes)
                    tokens = self._bpe_2(word_bytes)
                    for token in tokens:
                        token_ids.append(self.byte_to_id[token])
        return token_ids


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Stream-encode an iterable of string chunks (e.g. lines from a huge file)."""
        for chunk in iterable:
            token_ids = self.encode(chunk)
            yield from token_ids


    def decode(self, ids: list[int]) -> str:
        """Turn a list of token ids back into a string."""
        token_bytes = b''.join([self.vocab[token_id] for token_id in ids])
        return token_bytes.decode('utf-8', errors='replace')


    def _bpe_2(self, word_bytes: bytes) -> list[bytes]:
        """
        Algorithm 2 of applying BPE merge on a pre-token

        How it works:
        Iterate over the sequence,
        repeatedly finding the minimum-rank adjacent pair and applying just that one, until none apply.
        """
        # b'low' -> [b'l', b'o', b'w']
        word: list[bytes] = [bytes([b]) for b in word_bytes]

        if len(word) <= 1:
            return word
        
        while True:
            best_rank = float('inf')
            best_pair = None
            best_idx = -1
            i = 0

            while i < len(word) - 1:
                pair_id = self.merge_rank.get((word[i], word[i+1]), None)
                if pair_id is not None and pair_id < best_rank:
                    best_rank = pair_id
                    best_pair = (word[i], word[i+1])
                    best_idx = i
                i += 1

            if best_pair is None:
                break
            
            merged_bytes: list[bytes] = []
            j = 0
            while j < len(word):
                if j == best_idx:
                    merged_bytes.append(word[j] + word[j+1])
                    j += 2
                else:
                    merged_bytes.append(word[j])
                    j += 1
            word = merged_bytes
        
        return word

    def _bpe_1(self, word_bytes: bytes) -> list[bytes]:
        """
        Algorithm 1 of applying BPE merge on a pre-token
        
        How it works:
        Iterate over the merge list in order
        For each merge scan the current token sequence and apply it everywhere it appears. 

        Complexity: O(M * L) 
        where M is len(merges) and L is len(word_bytes) and M >> L
        """
        # b'low' -> [b'l', b'o', b'w']
        word: list[bytes] = [bytes([b]) for b in word_bytes]

        if len(word) <= 1:
            return word

        for pair in self.merges:
            merged_bytes: list[bytes] = []
            i = 0
            while i < len(word):
                if i + 1 < len(word) and word[i] == pair[0] and word[i+1] == pair[1]:
                    merged_bytes.append(word[i]+word[i+1])
                    i += 2
                else:
                    merged_bytes.append(word[i])
                    i += 1
            word = merged_bytes

        return word

        
from __future__ import annotations
from unittest import doModuleCleanups
from jaxtyping import Bool, Float, Int

import math
import torch
from torch import Tensor, nn
from einops import einsum, rearrange
from cs336_basics.nn_utils import softmax

class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # W: (d_out, d_in)
        self.weight = nn.Parameter(torch.empty((self.out_features, self.in_features), dtype=dtype, device=device))

        # Initialization: N(0, 2/(d_in + d_out)), truncated at [-3*sigma, 3*sigma]
        sigma = math.sqrt(2 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0, std=sigma, a=-3 * sigma, b=3 * sigma)

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:

        # Use einops as required in writeup
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # W: (vocab_size, d_model)
        self.weight = nn.Parameter(torch.empty((self.num_embeddings, self.embedding_dim), device=device, dtype=dtype))

        # Initialization: N(0, 1) truncated at [-3, 3]
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: Int[Tensor, ...]) -> Float[Tensor, "... d_model"]:

        # token_ids: (...) -> output: (..., d_model)
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization

    Given a vector a in R^{d_model}:
        RSM(a) = sqrt(mean(a**2) + eps)
        RMSNorm(a) = (a / RMS(a)) * g
    """

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        # W: (d_model, )
        # Initialization: 1
        self.weight = nn.Parameter(torch.ones((self.d_model,), device=device, dtype=dtype))

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        # Prevent overflow
        in_type = x.dtype
        x = x.to(torch.float32)

        # Perform RMSNorm
        # RMS(a): (..., 1)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        result = x / rms * self.weight

        return result.to(in_type)

def silu(x: Float[Tensor, ...]) -> Float[Tensor, ...]:
    return x * torch.sigmoid(x)

class SiLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        # W_1: (d_ff, d_model)
        # W_2: (d_model, d_ff)
        self.w1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        return self.w2(silu(self.w1(x)))

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        if d_ff is not None:
            self.d_ff = d_ff
        else:
            # Round (8/3 * d_model) to a nearby multiple of 64
            self.d_ff = 64 * round((d_model * 8 / 3) / 64)

        # W_1, W_3: (d_ff, d_model)
        # W_2: (d_model, d_ff)
        self.w1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)
        self.w3 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)


    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        y1 = silu(self.w1(x))
        y3 = self.w3(x)
        return self.w2(y1 * y3)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_k = d_k
        self.theta = theta

        # Positions [0, ..., max_len_seq-1]
        positions = torch.arange(self.max_seq_len, device=device, dtype=torch.float32)

        # [0, 2, 4, ..., d_k] so we don't need (2k-2)
        pair_idx = torch.arange(0, self.d_k, 2, device=device, dtype=torch.float32)
        frequencies = 1 / self.theta ** (pair_idx / self.d_k)

        # 1-D tensor t of shpae (N,):
        # t[:, None] gives a column (N, 1)
        # t[None, :] gives a row (1, N)
        angles = positions[:, None] * frequencies[None, :]  # (max_len_seq, d_k/2)

        cos = torch.cos(angles)  # (max_seq_len, d_k/2)
        sin = torch.sin(angles)  # (max_seq_len, d_k/2)

        self.register_buffer(name="cos", tensor=cos, persistent=False)
        self.register_buffer(name="sin", tensor=sin, persistent=False)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_k"],
        token_positions: Int[Tensor, "... seq_len"],
    ) -> Float[Tensor, "... seq_len d_k"]:

        cos = self.cos[token_positions]  # (max_seq_len, d_k/2) -> (..., seq_len, d_k/2)
        sin = self.sin[token_positions]

        in_type = x.dtype
        x = x.to(dtype=torch.float32)
        x_even = x[..., 0::2]  # (..., seq_len, d_k) -> (..., seq_len, d_k/2)
        x_odd = x[..., 1::2]

        x_even_rotated = cos * x_even - sin * x_odd
        x_odd_rotated = sin * x_even + cos * x_odd

        x_rotated = torch.stack(
            [x_even_rotated, x_odd_rotated], dim=-1
        )  # (..., seq_len, d_k/2) -> (..., seq_len, d_k/2, 2)
        x_rotated = rearrange(x_rotated, "... a b -> ... (a b)")  # (..., seq_len, d_k/2, 2) -> (..., seq_len, d_k)

        return x_rotated.to(dtype=in_type)


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:

    d_k = Q.shape[-1]
    
    score = einsum(Q, K, "... q d_k, ... k d_k -> ... q k") / math.sqrt(d_k)
    if mask is not None:
        score.masked_fill_(~mask, float("-inf"))
        
    attention = einsum(softmax(score, dim=-1), V, "... q k, ... k d_v -> ... q d_v")

    return attention

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self, 
        d_model:int, 
        num_heads:int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None, 
        rope: RotaryPositionalEmbedding | None = None,
        ablation: str | None = None
    ):
        super().__init__()

        self.ablation=ablation

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_v = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.rope = rope

    def forward(
        self, 
        x: Float[Tensor, " ... sequence_length d_model"],
        token_positions: Int[Tensor, "... seq_len"] | None = None
    ) -> Tensor:
        # Head is a batch dimension, we care about T and d_k
        Q = rearrange(self.q_proj(x), "... T (H d_k) -> ... H T d_k", H=self.num_heads)
        K = rearrange(self.k_proj(x), "... T (H d_k) -> ... H T d_k", H=self.num_heads)
        V = rearrange(self.v_proj(x), "... T (H d_v) -> ... H T d_v", H=self.num_heads)
        T = x.size(-2)

        # Apply RoPE (Q and K only)
        if self.ablation != "nope" and self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(T, dtype=torch.int, device=x.device)
            token_positions = rearrange(token_positions, "... T -> ... 1 T")
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # Causal masking - query i cannot attend to key j for all j > i
        mask = torch.tril(torch.ones((T, T), device=x.device, dtype=torch.bool))

        attention = scaled_dot_product_attention(Q, K, V, mask)

        attention = rearrange(attention, "... H T d_v -> ... T (H d_v)")

        return self.output_proj(attention)

class TransformerBlock(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None, 
        rope: RotaryPositionalEmbedding | None = None,
        ablation: str | None = None
    ) -> None:
        super().__init__()

        self.ablation=ablation

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, device=device, dtype=dtype, rope=rope, ablation=self.ablation)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        if self.ablation == "silu":
            self.ffn = SiLU(d_model, d_ff, device, dtype)
        else:
            self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self, 
        x: Float[Tensor, " batch sequence_length d_model"],
        token_positions: Int[Tensor, "... seq_len"] | None = None
    ) -> Float[Tensor, " batch sequence_length d_model"]:
        if self.ablation == "post_norm":
            x = self.ln1(x + self.attn(x, token_positions))
            x = self.ln2(x + self.ffn(x))
            return x

        if self.ablation == "no_norm":
            x = x + self.attn(x, token_positions)
            x = x + self.ffn(x)
            return x
        
        # Pre-norm
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int, 
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None,
        ablation: str | None = None
    ) -> None:
        super().__init__()
        
        # For 7.3 ablation
        self.ablation = ablation

        # Model size
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model

        # Embedding layer: (batch, seq_len,) -> (batch, seq_len, d_model)
        self.token_embeddings = Embedding(self.vocab_size, self.d_model, device=device, dtype=dtype)
        
        # RoPE is shared across blocks
        self.rope = RotaryPositionalEmbedding(
            theta=rope_theta, 
            d_k=d_model//num_heads,
            max_seq_len=self.context_length,
            device=device
        )

        # Transformer blocks: (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                device=device,
                dtype=dtype,
                rope=self.rope,
                ablation=self.ablation
            )
            for _ in range(num_layers)
        ])

        # Final normalization before LM Head
        self.ln_final = RMSNorm(self.d_model, device=device, dtype=dtype)

        # Final linear layer: (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)

    def forward(
        self,
        in_indices: Int[Tensor, " batch_size sequence_length"]
    ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        
        x = self.token_embeddings(in_indices)

        for block in self.layers:
            x = block(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits

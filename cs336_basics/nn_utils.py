from __future__ import annotations
from jaxtyping import Bool, Float, Int
from collections.abc import Iterable

import torch
from torch import Tensor


def softmax(x: Float[Tensor, "*shape"], dim: int) -> Float[Tensor, "*shape"]:
    c = torch.max(x, dim=dim, keepdim=True).values
    exponent = torch.exp(x - c)
    return exponent / exponent.sum(dim=dim, keepdim=True)

def cross_entropy(
    logits: Float[Tensor, " ... vocab_size"],  targets: Int[Tensor, " ..."]
) -> Float[Tensor, ""]:
    # Cast to float32 for mix-precision training
    logits = logits.to(torch.float32)

    # Substract max for numerical stability
    m = torch.max(logits, dim=-1, keepdim=True).values  # (..., 1)
    shifted = logits - m

    # Gather the shifted logits at target index (the True class)
    target_logits = torch.gather(shifted, dim=-1, index=targets.unsqueeze(-1)) # (..., 1)

    # Calculate logexpsum
    lse = torch.log(torch.sum(torch.exp(shifted), dim=-1, keepdim=True)) # (..., 1)

    # Negative log-likelihood
    nll = -(target_logits - lse)

    return torch.mean(nll)

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    grads = []

    for p in parameters:
        if p is None:
            continue
        g = p.grad
        if g is None:
            continue
        grads.append(g.reshape(-1))

    grad_l2_norm = torch.linalg.vector_norm(torch.cat(grads))

    if grad_l2_norm > max_l2_norm:
        for g in grads:
            with torch.no_grad():
                factor = max_l2_norm / (grad_l2_norm + 1e-6)
                g.mul_(factor)
    
    return grad_l2_norm
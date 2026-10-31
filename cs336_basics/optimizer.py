from math import sqrt, cos, pi
import torch
from collections.abc import Callable, Iterable
from typing import Optional

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, weight_decay, eps=1e-8) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group['betas']
            eps = group["eps"]
            weight_decay = group['weight_decay']
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p] # Get state associated with p
                grad = p.grad.data # Get the gradient of loss w.r.t. p
                m = state.get("m", torch.zeros_like(p)) # Get first moment vector or p-shaped 0s
                v = state.get("v", torch.zeros_like(p)) # Get second moment vector or p-shaped 0s

                t = state.get("t", 1) # Get iteration number from the state, or 1
                lr_t = lr * sqrt(1-(b2)**t) / (1-(b1)**t) # Compute adjusted lr for iteration t
                p.data -= lr * weight_decay * p.data # Apply weight decay
                m = b1 * m + (1-b1) * grad # Update the first moment estimate
                v = b2 * v + (1-b2) * grad**2 # Update the second moment estimate
                p.data -= lr_t * m / (torch.sqrt(v) + eps) # Appy moment-adjusted weight updates

                state["m"] = m # Update m in state
                state["v"] = v # Update v in state
                state["t"] = t + 1 # Update t in state

        return loss

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return it/warmup_iters * max_learning_rate
    
    if it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (1 + cos((it - warmup_iters) / (cosine_cycle_iters-warmup_iters) * pi)) * (max_learning_rate - min_learning_rate)
    
    return min_learning_rate
import json
import os
import argparse
import torch
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.nn_utils import softmax
from cs336_basics.training import load_checkpoint

def top_p_sampling(prob: torch.Tensor, top_p: float):
    sorted_prob, sorted_idx = torch.sort(prob, descending=True)
    cum_prob = torch.cumsum(sorted_prob, dim=-1)

    mask = cum_prob > top_p
    # Example with sorted probabilities [0.5, 0.3, 0.2] and p = 0.6: 
    # [0.5, 0.8, 1.0] -> [F, T, T] -> [F, F, T]
    # So 0.5 and 0.3 are kept
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False # Always keep the most likely token

    sorted_prob[mask] = 0.0
    prob.scatter_(dim=-1, index=sorted_idx, src=sorted_prob)

    return prob

@torch.no_grad()
def decode(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    eot_id,
    temperature: float = 1.0,
    top_p: float = 1.0,
):

    model.eval()
    context_length = getattr(model, "context_length", None)

    for _ in range(max_new_tokens):
        if context_length is not None and prompt_ids.numel() > context_length:
            context_window = prompt_ids[-context_length:]
        else:
            context_window = prompt_ids

        logits = model(context_window.unsqueeze(0)) # (1, W) -> (1, W, V)
        next_logits = logits[0, -1, :].to(torch.float32) # (V,)
        
        # Another design choice:
        # temperature = max(temperature, 1e-5)
        # Here we choose greedy decoding if temp is 0
        if temperature == 0:
            next_token_id = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            if temperature < 0:
                raise ValueError("Temperature must be non-negative")
            
            scaled_logits = next_logits / temperature
            prob = softmax(scaled_logits, dim=-1)

            if top_p < 1.0:
                prob = top_p_sampling(prob, top_p)
            next_token_id = torch.multinomial(prob, num_samples=1)
        
        prompt_ids = torch.cat([prompt_ids, next_token_id], dim=0)
        if next_token_id == eot_id:
            break
    return prompt_ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--run_path", type=str, required=True)
    parser.add_argument("--tokenizer", choices=['ts', 'owt'], default='ts')
    args = parser.parse_args()
    
    if not os.path.exists(args.run_path):
        raise ValueError(f"Run path {args.run_path} not found")
    else:
        config_path = os.path.join(args.run_path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
    
    if args.tokenizer == 'ts':
        vocab_path = "outputs/bpe/bpe_vocab_tinystories_10000_train_bpe.pkl"
        merge_path = "outputs/bpe/bpe_merges_tinystories_10000_train_bpe.pkl"
    else:
        vocab_path = "outputs/bpe/bpe_vocab_owt_10000_train_bpe.pkl"
        merge_path = "outputs/bpe/bpe_merges_owt_10000_train_bpe.pkl"     

    EOT = "<|endoftext|>"
    tokenizer = Tokenizer.from_files(
        vocab_path,
        merge_path,
        special_tokens=[EOT]
    )
    eot_id = tokenizer.byte_to_id[EOT.encode("utf-8")]

    model_dtype = torch.bfloat16 if config['dtype'] == 'bfloat16' else torch.float32
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = TransformerLM(
        vocab_size=config['vocab_size'],
        context_length=config['context_length'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config["num_heads"],
        d_ff=config['d_ff'],
        rope_theta=config['rope_theta'],
        device=device,
        dtype=model_dtype
    )

    ckpt_path = os.path.join(args.run_path, "checkpoints/final.pt")
    it = load_checkpoint(ckpt_path, model)

    if not args.prompt:
        input_prompt_ids = torch.tensor(tokenizer.encode(EOT), device=device)
    else:
        input_prompt_ids = torch.tensor(tokenizer.encode(args.prompt), device=device)
    output_token_ids = decode(
        model=model,
        prompt_ids=input_prompt_ids,
        max_new_tokens=args.max_new_tokens,
        eot_id=eot_id,
        temperature=args.temperature,
        top_p=args.top_p
    )

    print(tokenizer.decode(output_token_ids.tolist()))




def round_to_nearest(n, base):
    return base * round(n / base)

def calculate_flops(
    vocab_size,
    d_model,
    seq_len,
    num_layers,
    batch_size = 1
):
    d_ff = round_to_nearest(8/3 * d_model, 64)

    breakdown = {
        "attn_per_layer": {
            "proj": 8 * seq_len * d_model**2,
            "matmul": 4 * seq_len**2 * d_model
        },
        "ffn_per_layer" : 6 * seq_len * d_model * d_ff,
        "lm_head": 2 * seq_len * d_model * vocab_size
    }
    breakdown["attn_all_layers"] = {
        "proj": num_layers * breakdown["attn_per_layer"]['proj'],
        "matmul": num_layers * breakdown["attn_per_layer"]['matmul']
    }

    breakdown["attn_total"] = breakdown["attn_all_layers"]['proj'] + breakdown["attn_all_layers"]['matmul']
    breakdown["ffn_total"] = num_layers * breakdown["ffn_per_layer"]

    total = batch_size * (breakdown["attn_total"] + breakdown["ffn_total"] + breakdown["lm_head"])

    percentage = {
        "attn_proj": round(100 * breakdown["attn_all_layers"]['proj'] / total, 2),
        "attn_matmul": round(100 * breakdown["attn_all_layers"]['matmul'] / total, 2),
        "ffn": round(100 * breakdown["ffn_total"] / total, 2),
        "lm_head": round(100 * breakdown["lm_head"] / total, 2)
    }

    return total, breakdown, percentage
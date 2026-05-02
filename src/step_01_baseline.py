"""
Step 1: Verify GPT-2 Small performs the IOI task and do logit attribution.

Before studying the circuit, we need to confirm:
  1. The model actually predicts the IO name over the S name
  2. How strongly it does so (logit difference: logit(IO) - logit(S))
  3. Whether performance differs for common vs rare names

We also compute direct logit attribution per head: which heads contribute
most to the correct answer at the final token position?

Outputs:
    results/data/baseline_performance.json
    results/data/logit_attribution.json
"""

import torch
import numpy as np
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import test_prompt

from config import get_config, generate_ioi_dataset, save_results


def measure_ioi_performance(model, dataset, config):
    """
    For each prompt, measure:
      - logit(IO) - logit(S) at the final token position
      - whether the model's top prediction is the IO name
    """
    results = []

    for item in tqdm(dataset, desc="Measuring IOI performance"):
        prompt = item["prompt"]
        io_name = item["io_name"]
        s_name = item["s_name"]

        tokens = model.to_tokens(prompt, prepend_bos=True)

        with torch.no_grad():
            logits = model(tokens)  # (1, seq_len, vocab_size)

        # Logits at the final token position
        final_logits = logits[0, -1, :]  # (vocab_size,)

        # Get token IDs for the names
        # Names may be tokenized differently; take the first token of each
        io_token = model.to_tokens(" " + io_name, prepend_bos=False)[0, 0]
        s_token = model.to_tokens(" " + s_name, prepend_bos=False)[0, 0]

        io_logit = final_logits[io_token].item()
        s_logit = final_logits[s_token].item()
        logit_diff = io_logit - s_logit

        top_token = final_logits.argmax().item()
        top_prediction = model.to_string(torch.tensor([top_token]))
        is_correct = (top_token == io_token.item())

        results.append({
            "prompt": prompt,
            "io_name": io_name,
            "s_name": s_name,
            "io_logit": io_logit,
            "s_logit": s_logit,
            "logit_diff": logit_diff,
            "is_correct": is_correct,
            "top_prediction": top_prediction.strip(),
            "template_type": item["template_type"],
            "name_group": item["name_group"],
        })

    return results


def compute_logit_attribution(model, dataset, config):
    """
    Direct logit attribution: decompose the logit difference into
    contributions from each attention head.

    For each head, we compute:
        head_contribution = (head_output @ W_U[io_token] - head_output @ W_U[s_token])

    where head_output is the output of that head at the final token position,
    and W_U is the unembedding matrix.

    This tells us which heads are "pushing" the model toward the IO name.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    all_attributions = np.zeros((n_layers, n_heads))
    n_valid = 0

    for item in tqdm(dataset[:50], desc="Logit attribution"):
        prompt = item["prompt"]
        io_name = item["io_name"]
        s_name = item["s_name"]

        tokens = model.to_tokens(prompt, prepend_bos=True)
        io_token = model.to_tokens(" " + io_name, prepend_bos=False)[0, 0]
        s_token = model.to_tokens(" " + s_name, prepend_bos=False)[0, 0]

        _, cache = model.run_with_cache(tokens, return_type=None)

        # Unembedding direction for the logit difference
        W_U = model.W_U  # (d_model, vocab_size)
        logit_dir = W_U[:, io_token] - W_U[:, s_token]  # (d_model,)

        for layer in range(n_layers):
            # hook_z is the head output before W_O: (batch, seq_len, n_heads, d_head)
            z = cache[f"blocks.{layer}.attn.hook_z"][0, -1, :, :]  # (n_heads, d_head)
            W_O = model.W_O[layer]  # (n_heads, d_head, d_model)

            for head in range(n_heads):
                head_residual = z[head] @ W_O[head]  # (d_model,)
                contribution = (head_residual @ logit_dir).item()
                all_attributions[layer, head] += contribution

        n_valid += 1

    if n_valid > 0:
        all_attributions /= n_valid

    return all_attributions


def main():
    config = get_config()
    print(f"IOI Circuit Analysis - Step 1: Baseline + Attribution")
    print(f"  model: {config.model_name}")
    print(f"  device: {config.device}\n")

    model = HookedTransformer.from_pretrained(
        config.model_name, device=config.device, dtype=config.dtype,
    )
    model.eval()

    # Generate datasets
    print("Generating IOI prompts...")
    mixed_data = generate_ioi_dataset(config, name_group="mixed")
    common_data = generate_ioi_dataset(config, name_group="common")
    rare_data = generate_ioi_dataset(config, name_group="rare")

    # Baseline performance
    print("\n--- Baseline performance (mixed names) ---")
    mixed_results = measure_ioi_performance(model, mixed_data, config)
    accuracy = np.mean([r["is_correct"] for r in mixed_results])
    mean_logit_diff = np.mean([r["logit_diff"] for r in mixed_results])
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Mean logit diff (IO - S): {mean_logit_diff:.2f}")

    print("\n--- Common names ---")
    common_results = measure_ioi_performance(model, common_data, config)
    common_acc = np.mean([r["is_correct"] for r in common_results])
    common_ld = np.mean([r["logit_diff"] for r in common_results])
    print(f"  Accuracy: {common_acc:.1%}")
    print(f"  Mean logit diff: {common_ld:.2f}")

    print("\n--- Rare names ---")
    rare_results = measure_ioi_performance(model, rare_data, config)
    rare_acc = np.mean([r["is_correct"] for r in rare_results])
    rare_ld = np.mean([r["logit_diff"] for r in rare_results])
    print(f"  Accuracy: {rare_acc:.1%}")
    print(f"  Mean logit diff: {rare_ld:.2f}")

    save_results({
        "mixed": {
            "accuracy": accuracy,
            "mean_logit_diff": mean_logit_diff,
            "n_prompts": len(mixed_results),
            "per_prompt": mixed_results,
        },
        "common": {
            "accuracy": common_acc,
            "mean_logit_diff": common_ld,
            "n_prompts": len(common_results),
        },
        "rare": {
            "accuracy": rare_acc,
            "mean_logit_diff": rare_ld,
            "n_prompts": len(rare_results),
        },
    }, "baseline_performance.json", config)

    # Logit attribution
    print("\n--- Direct logit attribution ---")
    attributions = compute_logit_attribution(model, mixed_data, config)

    # Report top contributing heads
    flat = [(attributions[l, h], l, h)
            for l in range(attributions.shape[0])
            for h in range(attributions.shape[1])]
    flat.sort(reverse=True)

    print("  Top 10 heads by logit attribution (IO - S direction):")
    for val, l, h in flat[:10]:
        print(f"    L{l}H{h}: {val:+.3f}")

    print("\n  Bottom 5 (heads pushing toward S, i.e. negative name movers):")
    for val, l, h in flat[-5:]:
        print(f"    L{l}H{h}: {val:+.3f}")

    save_results({
        "attributions": attributions.tolist(),
        "top_10": [{"layer": l, "head": h, "value": v} for v, l, h in flat[:10]],
        "bottom_5": [{"layer": l, "head": h, "value": v} for v, l, h in flat[-5:]],
    }, "logit_attribution.json", config)

    print("\nStep 1 done. Next: python src/step_02_patching.py")


if __name__ == "__main__":
    main()
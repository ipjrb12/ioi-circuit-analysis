"""
Step 2: Activation patching to identify the IOI circuit.

Activation patching (causal scrubbing lite): for each attention head, replace
its output with the output from a corrupted prompt where the IO and S names
are swapped. If patching a head causes a large drop in logit difference,
that head is part of the IOI circuit.

We also do path patching through the residual stream at each layer to get
a layer-level view of where information flows.

Corrupted prompt: swap IO and S names. E.g.:
    Clean:   "When Mary and John went to the store, John gave a drink to" -> Mary
    Corrupt: "When John and Mary went to the store, Mary gave a drink to" -> John

Outputs:
    results/data/head_patching.json
    results/data/layer_patching.json
"""

import torch
import numpy as np
from tqdm import tqdm
from transformer_lens import HookedTransformer

from config import get_config, generate_ioi_dataset, save_results


def make_corrupted_prompt(item):
    """Swap IO and S names to create the corrupted version."""
    prompt = item["prompt"]
    io_name = item["io_name"]
    s_name = item["s_name"]

    # Replace names with placeholders first to avoid double-replacement
    corrupted = prompt.replace(io_name, "<<IO>>").replace(s_name, "<<S>>")
    corrupted = corrupted.replace("<<IO>>", s_name).replace("<<S>>", io_name)

    return corrupted


def get_logit_diff(model, tokens, io_token, s_token):
    """Compute logit(IO) - logit(S) at the final position."""
    with torch.no_grad():
        logits = model(tokens)
    final_logits = logits[0, -1, :]
    return (final_logits[io_token] - final_logits[s_token]).item()


def run_head_patching(model, dataset, config):
    """
    For each attention head: run the clean prompt, but replace that head's
    output with its output on the corrupted prompt. Measure how much the
    logit difference drops.

    A large drop = the head is important for the IOI task.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Patching effect: how much logit_diff changes when we patch each head
    patching_effects = np.zeros((n_layers, n_heads))
    n_valid = 0

    for item in tqdm(dataset[:50], desc="Head patching"):
        clean_prompt = item["prompt"]
        corrupted_prompt = make_corrupted_prompt(item)

        clean_tokens = model.to_tokens(clean_prompt, prepend_bos=True)
        corrupt_tokens = model.to_tokens(corrupted_prompt, prepend_bos=True)

        io_token = model.to_tokens(" " + item["io_name"], prepend_bos=False)[0, 0]
        s_token = model.to_tokens(" " + item["s_name"], prepend_bos=False)[0, 0]

        # Verify tokens align (skip if they don't due to different tokenization)
        if clean_tokens.shape != corrupt_tokens.shape:
            continue

        # Get clean logit diff
        clean_ld = get_logit_diff(model, clean_tokens, io_token, s_token)

        # Cache corrupt activations
        _, corrupt_cache = model.run_with_cache(corrupt_tokens, return_type=None)

        # Patch each head one at a time
        for layer in range(n_layers):
            for head in range(n_heads):
                hook_name = f"blocks.{layer}.attn.hook_z"
                corrupt_head_out = corrupt_cache[hook_name][0, :, head, :]  # (seq_len, d_head)

                def patch_hook(z, hook, _head=head, _corrupt=corrupt_head_out):
                    z[0, :, _head, :] = _corrupt.to(z.device)
                    return z

                patched_ld = model.run_with_hooks(
                    clean_tokens,
                    fwd_hooks=[(hook_name, patch_hook)],
                    return_type="logits",
                )[0, -1, :]
                patched_ld = (patched_ld[io_token] - patched_ld[s_token]).item()

                # Effect = clean - patched (positive means head was helping)
                patching_effects[layer, head] += (clean_ld - patched_ld)

        n_valid += 1

    if n_valid > 0:
        patching_effects /= n_valid

    return patching_effects


def run_layer_patching(model, dataset, config):
    """
    Patch the entire residual stream at each layer position.
    Gives a coarser view of where information flows.
    """
    n_layers = model.cfg.n_layers
    layer_effects = np.zeros(n_layers)
    n_valid = 0

    for item in tqdm(dataset[:50], desc="Layer patching"):
        clean_prompt = item["prompt"]
        corrupted_prompt = make_corrupted_prompt(item)

        clean_tokens = model.to_tokens(clean_prompt, prepend_bos=True)
        corrupt_tokens = model.to_tokens(corrupted_prompt, prepend_bos=True)

        if clean_tokens.shape != corrupt_tokens.shape:
            continue

        io_token = model.to_tokens(" " + item["io_name"], prepend_bos=False)[0, 0]
        s_token = model.to_tokens(" " + item["s_name"], prepend_bos=False)[0, 0]

        clean_ld = get_logit_diff(model, clean_tokens, io_token, s_token)

        _, corrupt_cache = model.run_with_cache(corrupt_tokens, return_type=None)

        for layer in range(n_layers):
            hook_name = f"blocks.{layer}.hook_resid_post"
            corrupt_resid = corrupt_cache[hook_name]

            def patch_hook(resid, hook, _corrupt=corrupt_resid):
                return _corrupt.to(resid.device)

            patched_logits = model.run_with_hooks(
                clean_tokens,
                fwd_hooks=[(hook_name, patch_hook)],
                return_type="logits",
            )
            patched_ld = (patched_logits[0, -1, io_token] - patched_logits[0, -1, s_token]).item()
            layer_effects[layer] += (clean_ld - patched_ld)

        n_valid += 1

    if n_valid > 0:
        layer_effects /= n_valid

    return layer_effects


def run_name_frequency_comparison(model, config):
    """
    Compare patching effects for common vs rare names.
    Does the circuit change depending on name frequency?
    """
    print("\n--- Common names ---")
    common_data = generate_ioi_dataset(config, name_group="common")
    common_effects = run_head_patching(model, common_data, config)

    print("\n--- Rare names ---")
    rare_data = generate_ioi_dataset(config, name_group="rare")
    rare_effects = run_head_patching(model, rare_data, config)

    return common_effects, rare_effects


def main():
    config = get_config()
    print(f"IOI Circuit Analysis - Step 2: Activation Patching")
    print(f"  model: {config.model_name}")
    print(f"  device: {config.device}\n")

    model = HookedTransformer.from_pretrained(
        config.model_name, device=config.device, dtype=config.dtype,
    )
    model.eval()

    dataset = generate_ioi_dataset(config, name_group="mixed")

    # Head-level patching
    print("--- Head patching (mixed names) ---")
    head_effects = run_head_patching(model, dataset, config)

    flat = [(head_effects[l, h], l, h)
            for l in range(head_effects.shape[0])
            for h in range(head_effects.shape[1])]
    flat.sort(reverse=True)

    print("\nTop 10 most important heads (patching causes largest LD drop):")
    for val, l, h in flat[:10]:
        print(f"  L{l}H{h}: {val:+.3f}")

    print("\nTop 5 negative movers (patching increases LD):")
    for val, l, h in flat[-5:]:
        print(f"  L{l}H{h}: {val:+.3f}")

    # Layer-level patching
    print("\n--- Layer patching ---")
    layer_effects = run_layer_patching(model, dataset, config)
    for layer, effect in enumerate(layer_effects):
        print(f"  Layer {layer}: {effect:+.3f}")

    # Name frequency comparison
    print("\n--- Name frequency comparison ---")
    common_effects, rare_effects = run_name_frequency_comparison(model, config)

    save_results({
        "head_effects_mixed": head_effects.tolist(),
        "layer_effects": layer_effects.tolist(),
        "head_effects_common": common_effects.tolist(),
        "head_effects_rare": rare_effects.tolist(),
        "top_10_heads": [{"layer": l, "head": h, "effect": v} for v, l, h in flat[:10]],
    }, "head_patching.json", config)

    print("\nStep 2 done. Next: python src/step_03_figures.py")


if __name__ == "__main__":
    main()
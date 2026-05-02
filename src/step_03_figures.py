"""
Step 3: Generate figures.

Figures:
    1. Head patching heatmap (layer x head) — the "circuit map"
    2. Direct logit attribution heatmap
    3. Layer-wise patching effect bar chart
    4. Common vs rare name comparison (scatter or side-by-side heatmaps)
    5. Baseline performance comparison (common vs rare)
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from config import get_config

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save(fig, name, config):
    fig.savefig(Path(config.figures_dir) / f"{name}.png")
    fig.savefig(Path(config.figures_dir) / f"{name}.pdf")
    plt.close(fig)
    print(f"  {name}.png")


def plot_head_patching_heatmap(data, config):
    """The key figure: which heads matter for the IOI task."""
    effects = np.array(data["head_effects_mixed"])
    n_layers, n_heads = effects.shape

    fig, ax = plt.subplots(figsize=(12, 5))
    vmax = max(abs(effects.min()), abs(effects.max()))
    im = ax.imshow(effects, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_layers))
    ax.set_title("Activation patching: effect on logit difference (IO - S)")

    # Annotate the strongest heads
    for l in range(n_layers):
        for h in range(n_heads):
            val = effects[l, h]
            if abs(val) > vmax * 0.3:
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(h, l, f"{val:.1f}", ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Patching effect (clean LD - patched LD)")
    _save(fig, "fig1_head_patching", config)


def plot_logit_attribution(data, config):
    """Which heads directly push logits toward IO vs S?"""
    attr = np.array(data["attributions"])
    n_layers, n_heads = attr.shape

    fig, ax = plt.subplots(figsize=(12, 5))
    vmax = max(abs(attr.min()), abs(attr.max()))
    im = ax.imshow(attr, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_layers))
    ax.set_title("Direct logit attribution: head contribution to IO - S logit")

    for l in range(n_layers):
        for h in range(n_heads):
            val = attr[l, h]
            if abs(val) > vmax * 0.3:
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(h, l, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Attribution (positive = promotes IO)")
    _save(fig, "fig2_logit_attribution", config)


def plot_layer_effects(data, config):
    """Layer-level patching effect."""
    effects = data["layer_effects"]
    layers = list(range(len(effects)))

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#e85d26" if e > 0 else "#3b82f6" for e in effects]
    ax.bar(layers, effects, color=colors, alpha=0.85)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Patching effect on logit difference")
    ax.set_title("Layer-wise activation patching: where does IOI information flow?")
    ax.set_xticks(layers)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

    _save(fig, "fig3_layer_patching", config)


def plot_name_frequency_comparison(data, config):
    """Side-by-side: do the same heads matter for common vs rare names?"""
    common = np.array(data["head_effects_common"])
    rare = np.array(data["head_effects_rare"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    vmax = max(abs(common).max(), abs(rare).max())

    for ax, mat, title in zip(axes, [common, rare], ["Common names", "Rare names"]):
        im = ax.imshow(mat, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_xticks(range(mat.shape[1]))
        ax.set_yticks(range(mat.shape[0]))
        ax.set_title(title)

    fig.suptitle("Does the IOI circuit generalize across name frequency?", fontsize=14,
                 fontweight="bold")
    plt.colorbar(im, ax=axes, label="Patching effect", shrink=0.8)
    plt.tight_layout()
    _save(fig, "fig4_name_frequency", config)


def plot_name_frequency_scatter(data, config):
    """Scatter: common vs rare patching effect per head."""
    common = np.array(data["head_effects_common"]).flatten()
    rare = np.array(data["head_effects_rare"]).flatten()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(common, rare, alpha=0.5, s=20)

    # Identity line
    lims = [min(common.min(), rare.min()), max(common.max(), rare.max())]
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)

    # Label outliers
    n_layers = np.array(data["head_effects_common"]).shape[0]
    n_heads = np.array(data["head_effects_common"]).shape[1]
    for i, (c, r) in enumerate(zip(common, rare)):
        if abs(c) > np.percentile(np.abs(common), 95) or abs(r) > np.percentile(np.abs(rare), 95):
            l, h = divmod(i, n_heads)
            ax.annotate(f"L{l}H{h}", (c, r), fontsize=7, alpha=0.7)

    corr = np.corrcoef(common, rare)[0, 1]
    ax.set_xlabel("Patching effect (common names)")
    ax.set_ylabel("Patching effect (rare names)")
    ax.set_title(f"Per-head patching: common vs rare names (r={corr:.2f})")

    _save(fig, "fig5_name_scatter", config)


def plot_baseline_comparison(data, config):
    """Bar chart comparing accuracy and logit diff for common vs rare."""
    categories = ["mixed", "common", "rare"]
    accuracies = [data[c]["accuracy"] for c in categories]
    logit_diffs = [data[c]["mean_logit_diff"] for c in categories]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(categories, accuracies, color=["#8b5cf6", "#3b82f6", "#e85d26"], alpha=0.85)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("IOI task accuracy")
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=10)

    axes[1].bar(categories, logit_diffs, color=["#8b5cf6", "#3b82f6", "#e85d26"], alpha=0.85)
    axes[1].set_ylabel("Mean logit diff (IO - S)")
    axes[1].set_title("IOI logit difference")
    for i, v in enumerate(logit_diffs):
        axes[1].text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=10)

    fig.suptitle("Baseline IOI performance by name frequency", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "fig6_baseline_comparison", config)


def main():
    config = get_config()
    Path(config.figures_dir).mkdir(parents=True, exist_ok=True)
    print("Generating figures...")

    dd = Path(config.data_dir)

    if (dd / "head_patching.json").exists():
        with open(dd / "head_patching.json") as f:
            patching_data = json.load(f)
        plot_head_patching_heatmap(patching_data, config)
        plot_layer_effects(patching_data, config)
        plot_name_frequency_comparison(patching_data, config)
        plot_name_frequency_scatter(patching_data, config)
    else:
        print("  head_patching.json not found, run step 02 first.")

    if (dd / "logit_attribution.json").exists():
        with open(dd / "logit_attribution.json") as f:
            plot_logit_attribution(json.load(f), config)
    else:
        print("  logit_attribution.json not found, run step 01 first.")

    if (dd / "baseline_performance.json").exists():
        with open(dd / "baseline_performance.json") as f:
            plot_baseline_comparison(json.load(f), config)
    else:
        print("  baseline_performance.json not found.")

    print("Done.")


if __name__ == "__main__":
    main()
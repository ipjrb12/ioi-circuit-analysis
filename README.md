# IOI Circuit Analysis in GPT-2 Small: Activation Patching and Name Frequency Generalization

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![TransformerLens](https://img.shields.io/badge/TransformerLens-1.x-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

Identifying the circuit for indirect object identification (IOI) in GPT-2 Small using activation patching and direct logit attribution, with an extension testing whether the circuit generalises across name frequencies.

## Research question

Which attention heads in GPT-2 Small implement the IOI task ("When Mary and John went to the store, John gave a drink to ___" → Mary), and does the same circuit handle common names (John, Mary) and rare names (Dmitri, Yuki)?

## Key findings

**1. The IOI circuit is concentrated in a small set of late-layer heads.** Activation patching identifies L9H9 (effect: 1.34), L8H10 (0.90), L10H0 (0.83), and L4H11 (0.74) as the most important positive contributors. L10H7 is the strongest negative mover (effect: −3.14), actively pushing toward the subject name — consistent with the S-inhibition heads described in Wang et al. (2022).

**2. Direct logit attribution reveals sharper concentration.** L9H9 contributes +39.4 to the IO−S logit difference, followed by L10H10 (+13.1) and L9H6 (+12.0). The negative movers are L10H7 (−23.3) and L11H10 (−14.6). Attribution magnitudes exceed patching effects because attribution measures direct logit contribution while patching measures causal effect of removal, which other heads partially compensate.

**3. The circuit generalises across name frequency (r = 0.87).** Per-head patching effects for common and rare names are highly correlated. The same heads — L9H9, L8H10, L7H9, L10H7 — appear as top contributors for both groups. The IOI circuit operates on syntactic structure (which entity appeared first, which was repeated) rather than memorised name associations.

**4. Performance degrades for rare names, but the circuit does not change.** Accuracy drops from 83% (common) to 65% (rare), and logit difference from 3.53 to 3.14. The model is less confident on rare names but uses the same computational pathway. The circuit generalises even as raw performance degrades.

## Results

### Head-level activation patching

![Head Patching](results/figures/fig1_head_patching.png)

Each cell shows how much the IO−S logit difference drops when that head's output is replaced with the corrupted (names-swapped) version. Red = important for correct prediction. Blue = pushes toward the wrong answer. The circuit is sparse: most heads have near-zero effect.

### Direct logit attribution

![Logit Attribution](results/figures/fig2_logit_attribution.png)

Each head's direct contribution to the IO−S logit at the final token. L9H9 dominates (+39.4), acting as the primary name-mover head. Late-layer negative heads (L10H7 at −23.3, L11H10 at −14.6) are S-inhibition mechanisms that suppress the subject name.

### Name frequency comparison

![Name Frequency](results/figures/fig4_name_frequency.png)

Side-by-side patching heatmaps for common and rare names. The circuit pattern is preserved across both conditions, with variation in magnitude but not structure.

### Per-head scatter: common vs rare

![Scatter](results/figures/fig5_name_scatter.png)

Each point is one attention head. The r = 0.87 correlation indicates strong generalisation. Key circuit heads cluster in the same relative positions for both name groups.

### Baseline performance

![Baseline](results/figures/fig6_baseline_comparison.png)

The model performs above chance for all name groups. Common names are easier (83% accuracy, logit diff 3.53) than rare names (65%, 3.14), consistent with greater pre-training exposure.

## Method

1. **Task.** Indirect Object Identification: "When [IO] and [S] went to the [place], [S] gave a [object] to ___" → model should predict IO. Five template variants, 10 common names (John, Mary, ...), 10 rare names (Dmitri, Yuki, ...), random places and objects.
2. **Activation patching.** For each of the 144 attention heads (12 layers × 12 heads): run the clean prompt, replace that head's output at `hook_z` with the corrupted (IO/S-swapped) version, measure the drop in IO−S logit difference.
3. **Direct logit attribution.** Decompose the IO−S logit difference into per-head contributions by projecting each head's output through the unembedding matrix in the IO−S direction.
4. **Name frequency comparison.** Run the full patching experiment separately on common-only and rare-only name sets. Compare per-head effects via Pearson correlation.

## Limitations

- **Layer patching was uninformative.** Patching the full residual stream at each layer gave uniform high effects (~6.77), likely because replacing the full residual at any layer destroys nearly all task-relevant information. Token-position-specific patching would be more targeted.
- **Template diversity.** Five templates with a fixed structure may not capture the full range of IOI constructions the model handles in practice.
- **No attention pattern analysis.** We identify which heads matter but don't show what they attend to. Attention visualisations would clarify the mechanistic story — for example, confirming that name-mover heads attend to the IO position.
- **Single metric.** We use logit difference as the primary measure. Adding probability-based or KL divergence metrics would give a fuller picture.

## Future work

- Add attention pattern visualisations to confirm the role of name-mover vs S-inhibition heads
- Extend to a second circuit task (e.g. greater-than or docstring) to test whether the same patching methodology generalises
- Use the token-position-specific patterns found here as a basis for the safety-relevant feature analysis in [sae-safety-features](https://github.com/ipjrb12/sae-safety-features), where understanding which positions carry safety-relevant signal matters

## Related work

- Wang et al. (2022). *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small.* Our results replicate their core name-mover finding at L9H9, L9H6, L10H0.
- Conmy et al. (2023). *Towards Automated Circuit Discovery for Mechanistic Interpretability.*
- Nanda (2023). *200 Concrete Open Problems in Mechanistic Interpretability.*

## Reproducing

```bash
pip install torch transformer_lens transformers einops jaxtyping numpy matplotlib tqdm

mkdir -p results/figures results/data
python src/step_01_baseline.py
python src/step_02_patching.py
python src/step_03_figures.py
```

## Project structure

```
├── README.md
├── src/
│   ├── config.py              # Configuration, IOI dataset generation
│   ├── step_01_baseline.py    # Task performance + logit attribution
│   ├── step_02_patching.py    # Activation patching (head + layer)
│   └── step_03_figures.py     # Figure generation
└── results/
    ├── data/
    │   ├── baseline_performance.json
    │   ├── logit_attribution.json
    │   └── head_patching.json
    └── figures/
        ├── fig1_head_patching.png
        ├── fig2_logit_attribution.png
        ├── fig4_name_frequency.png
        ├── fig5_name_scatter.png
        └── fig6_baseline_comparison.png
```

## Part of a research series

This is the first project in a series exploring how LLMs represent and process safety-relevant information.

| # | Repo | Question |
|---|------|----------|
| 1 | **ioi-circuit-analysis** (this repo) | Which circuits implement structured tasks in GPT-2? |
| 2 | [sae-safety-features](https://github.com/ipjrb12/sae-safety-features) | Do SAEs find distinct features for safety-relevant inputs? |
| 3 | [mats_jailbreak](https://github.com/ipjrb12/mats_jailbreak) | How do decomposed attacks exploit the refusal gap? |
| 4 | [LayerCrit-Jailbreak-Susceptibility](https://github.com/ipjrb12/LayerCrit-Jailbreak-Susceptibility) | Can jailbreak susceptibility be measured before generation? |
| 5 | [rm-probing-experiment](https://github.com/ipjrb12/rm-probing-experiment) | What do reward model internals encode about response quality? |
| 6 | [weak-to-strong-generalization](https://github.com/ipjrb12/weak-to-strong-generalization) | What happens when the supervisor's labels are too noisy? |

## Tools

- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
- PyTorch, matplotlib

## License

MIT

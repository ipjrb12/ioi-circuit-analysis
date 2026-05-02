"""
Configuration and dataset generation for the IOI circuit analysis.

We study the Indirect Object Identification (IOI) task:
    "When Mary and John went to the store, John gave a drink to"
    -> model should predict "Mary" (the indirect object)

The core question: which attention heads are responsible for this behavior,
and does the circuit generalize across name frequencies?
"""

import os
import sys
import torch
import json
import random
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Config:
    model_name: str = "gpt2-small"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

    # IOI dataset
    n_prompts: int = 100
    seed: int = 42

    # Activation patching
    batch_size: int = 10

    # Paths
    results_dir: str = "results"
    figures_dir: str = "results/figures"
    data_dir: str = "results/data"


def get_config():
    return Config()


# ---------------------------------------------------------------------------
# IOI prompt generation
# ---------------------------------------------------------------------------
# Template: "When [IO] and [S] went to the [place], [S] gave a [object] to"
# Correct answer: IO (the indirect object)
#
# We split names into "common" and "rare" to test generalization.

TEMPLATES = [
    "When {IO} and {S} went to the {place}, {S} gave a {object} to",
    "When {IO} and {S} arrived at the {place}, {S} handed a {object} to",
    "When {IO} and {S} walked to the {place}, {S} passed a {object} to",
    "When {IO} and {S} got to the {place}, {S} offered a {object} to",
    "When {IO} and {S} entered the {place}, {S} showed a {object} to",
]

# Reversed templates: S mentioned first, IO second
TEMPLATES_REVERSED = [
    "When {S} and {IO} went to the {place}, {S} gave a {object} to",
    "When {S} and {IO} arrived at the {place}, {S} handed a {object} to",
    "When {S} and {IO} walked to the {place}, {S} passed a {object} to",
]

COMMON_NAMES = [
    "John", "Mary", "James", "Sarah", "David",
    "Lisa", "Robert", "Emma", "Michael", "Anna",
]

RARE_NAMES = [
    "Dmitri", "Yuki", "Priya", "Sven", "Amara",
    "Reiko", "Bodhi", "Isolde", "Kenji", "Astrid",
]

PLACES = [
    "store", "park", "library", "restaurant", "office",
    "school", "market", "beach", "museum", "garden",
]

OBJECTS = [
    "drink", "book", "letter", "present", "key",
    "ticket", "flower", "phone", "bag", "note",
]


def generate_ioi_dataset(config, name_group="mixed"):
    """
    Generate IOI prompts. Returns list of dicts:
        {
            "prompt": str,
            "io_name": str,       # correct answer (indirect object)
            "s_name": str,        # subject
            "template_type": str, # "normal" or "reversed"
            "name_group": str,    # "common", "rare", or "mixed"
        }
    """
    random.seed(config.seed)

    if name_group == "common":
        names = COMMON_NAMES
    elif name_group == "rare":
        names = RARE_NAMES
    else:
        names = COMMON_NAMES + RARE_NAMES

    dataset = []
    for _ in range(config.n_prompts):
        io_name, s_name = random.sample(names, 2)
        place = random.choice(PLACES)
        obj = random.choice(OBJECTS)

        # Mix normal and reversed templates
        if random.random() < 0.5:
            template = random.choice(TEMPLATES)
            ttype = "normal"
        else:
            template = random.choice(TEMPLATES_REVERSED)
            ttype = "reversed"

        prompt = template.format(IO=io_name, S=s_name, place=place, object=obj)

        dataset.append({
            "prompt": prompt,
            "io_name": io_name,
            "s_name": s_name,
            "template_type": ttype,
            "name_group": name_group,
        })

    return dataset


def save_results(data, filename, config):
    path = Path(config.data_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved {path}")


def load_results(filename, config):
    with open(Path(config.data_dir) / filename) as f:
        return json.load(f)
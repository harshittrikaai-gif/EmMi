"""
Stratified data sampling for tokenizer training.

Upsamples low-resource languages to ensure adequate representation in
the learned vocabulary.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

from emmit.tokenizer.config import LANGUAGE_GROUPS


# ---------------------------------------------------------------------------
# Sampling weights per group
# ---------------------------------------------------------------------------

DEFAULT_SAMPLING_STRATEGY = {
    "high_resource": 0.50,
    "medium_resource": 0.35,
    "low_resource": 0.15,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare_tokenizer_training_data(
    data_sources: Dict[str, List[str]],
    target_samples: int = 10_000_000,
    sampling_strategy: Optional[Dict[str, float]] = None,
    seed: int = 42,
) -> List[str]:
    """
    Sample documents across language groups for tokenizer training.

    Args:
        data_sources:      ``{lang_code: [list of text documents]}``
        target_samples:    total number of text chunks to return
        sampling_strategy: weight per language group (default: 50/35/15)
        seed:              random seed for reproducibility

    Returns:
        Shuffled list of sampled text chunks.
    """
    if sampling_strategy is None:
        sampling_strategy = DEFAULT_SAMPLING_STRATEGY

    rng = random.Random(seed)
    sampled: List[str] = []

    for group_name, weight in sampling_strategy.items():
        group_langs = LANGUAGE_GROUPS.get(group_name, [])
        group_budget = int(target_samples * weight)

        if not group_langs:
            continue

        per_lang_budget = group_budget // len(group_langs)

        for lang in group_langs:
            docs = data_sources.get(lang, [])
            if not docs:
                continue

            # Sample with replacement if corpus is smaller than budget
            n = min(per_lang_budget, len(docs))
            if n < per_lang_budget:
                chosen = rng.choices(docs, k=per_lang_budget)
            else:
                chosen = rng.sample(docs, n)

            sampled.extend(chosen)

    rng.shuffle(sampled)
    return sampled


def save_samples_to_file(samples: List[str], output_path: str | Path) -> None:
    """Write sampled texts to a plain-text file (one document per line)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in samples:
            # Replace newlines within a doc so each line = one doc
            f.write(doc.replace("\n", " ").strip() + "\n")

"""
Multilingual tokenizer configuration.

Defines vocabulary allocation, special tokens, and SentencePiece training
settings for 52 languages (22 Indian + 30 global).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# SentencePiece training configuration
# ---------------------------------------------------------------------------

TOKENIZER_CONFIG = {
    "model_type": "unigram",
    "vocab_size": 128000,
    "character_coverage": 0.9995,
    "normalization_rule_name": "nmt_nfkc",

    # Token conventions
    "unk_token": "<unk>",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",

    # Multilingual handling
    "byte_fallback": True,
    "split_by_whitespace": True,
    "split_by_unicode_script": True,

    # Additional special tokens (vision + language tags)
    "additional_special_tokens": [
        # Vision
        "<|image|>",
        "<|endofimage|>",
        # Indian languages
        "<|hindi|>", "<|bengali|>", "<|tamil|>", "<|telugu|>",
        "<|marathi|>", "<|gujarati|>", "<|kannada|>", "<|malayalam|>",
        "<|odia|>", "<|punjabi|>", "<|assamese|>", "<|urdu|>",
        "<|maithili|>", "<|sanskrit|>", "<|konkani|>", "<|nepali|>",
        "<|sindhi|>", "<|dogri|>", "<|manipuri|>", "<|bodo|>",
        "<|santali|>", "<|kashmiri|>",
        # Global languages
        "<|english|>", "<|spanish|>", "<|french|>", "<|german|>",
        "<|chinese|>", "<|japanese|>", "<|korean|>", "<|arabic|>",
        "<|russian|>", "<|portuguese|>", "<|italian|>", "<|dutch|>",
        "<|turkish|>", "<|polish|>", "<|swedish|>",
    ],
}

# ---------------------------------------------------------------------------
# Vocabulary allocation across scripts (target distribution)
# ---------------------------------------------------------------------------

VOCAB_ALLOCATION = {
    "latin": 40_000,
    "devanagari": 25_000,
    "bengali": 10_000,
    "tamil": 8_000,
    "telugu": 8_000,
    "kannada": 6_000,
    "malayalam": 6_000,
    "gujarati": 5_000,
    "gurmukhi": 4_000,
    "odia": 4_000,
    "cjk": 20_000,
    "arabic": 8_000,
    "other": 4_000,
}

# ---------------------------------------------------------------------------
# Language metadata
# ---------------------------------------------------------------------------

INDIC_LANGUAGES = {
    "hi", "bn", "te", "mr", "ta", "ur", "gu", "kn", "or", "ml",
    "pa", "as", "mai", "sa", "kok", "ne", "sd", "doi", "mni", "brx",
    "sat", "ks",
}

LANGUAGE_SCRIPT_MAP = {
    "hi": "Devanagari",
    "mr": "Devanagari",
    "sa": "Devanagari",
    "ne": "Devanagari",
    "kok": "Devanagari",
    "mai": "Devanagari",
    "doi": "Devanagari",
    "bn": "Bengali",
    "as": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "gu": "Gujarati",
    "pa": "Gurmukhi",
    "or": "Odia",
    "ur": "Arabic",
    "sd": "Arabic",
    "ks": "Arabic",
    "mni": "Meetei Mayek",
    "sat": "Ol Chiki",
    "brx": "Devanagari",
}

# Language groups for sampling
LANGUAGE_GROUPS = {
    "high_resource": ["en", "hi", "es", "fr", "de", "zh", "ja"],
    "medium_resource": ["ta", "te", "bn", "mr", "gu", "kn", "ml"],
    "low_resource": ["as", "or", "mai", "sa", "kok", "mni", "sat"],
}

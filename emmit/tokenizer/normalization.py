"""
Script-specific text normalization for Indic languages.

Handles Unicode normalization, nukta/halant normalisation for Devanagari,
vowel-sign normalisation for Dravidian scripts, and zero-width character
cleanup.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

from emmit.tokenizer.config import LANGUAGE_SCRIPT_MAP


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_indic_text(text: str, lang_code: str) -> str:
    """
    Apply script-specific normalization for Indic languages.

    Args:
        text:      raw text string
        lang_code: ISO 639-1 language code (e.g. ``"hi"``, ``"ta"``)

    Returns:
        normalised text
    """
    # NFKC normalization (compatibility decomposition + canonical composition)
    text = unicodedata.normalize("NFKC", text)

    # Language / script-specific rules
    script = LANGUAGE_SCRIPT_MAP.get(lang_code)

    if script == "Devanagari":
        text = normalize_nukta(text)
        text = normalize_halant(text)
    elif script in ("Tamil", "Telugu", "Kannada", "Malayalam"):
        text = normalize_vowel_signs(text)

    # Clean zero-width characters (keep ZWNJ / ZWJ which are meaningful)
    text = remove_zero_width(text)

    return text


# ---------------------------------------------------------------------------
# Devanagari helpers
# ---------------------------------------------------------------------------

# Pre-composed nukta characters → canonical representations
_NUKTA_MAP = {
    "\u0958": "\u0915\u093C",  # क़
    "\u0959": "\u0916\u093C",  # ख़
    "\u095A": "\u0917\u093C",  # ग़
    "\u095B": "\u091C\u093C",  # ज़
    "\u095C": "\u0921\u093C",  # ड़
    "\u095D": "\u0922\u093C",  # ढ़
    "\u095E": "\u092B\u093C",  # फ़
    "\u095F": "\u092F\u093C",  # य़
}


def normalize_nukta(text: str) -> str:
    """Replace pre-composed nukta characters with base + nukta sequences."""
    for composed, decomposed in _NUKTA_MAP.items():
        text = text.replace(composed, decomposed)
    return text


def normalize_halant(text: str) -> str:
    """
    Normalise halant (virama) usage.

    Removes duplicate halants and strips trailing halant + whitespace
    sequences that are likely noise.
    """
    # Remove duplicate halant (U+094D)
    text = re.sub("\u094D{2,}", "\u094D", text)
    return text


# ---------------------------------------------------------------------------
# Dravidian helpers
# ---------------------------------------------------------------------------

def normalize_vowel_signs(text: str) -> str:
    """
    Normalise vowel sign ordering for Dravidian scripts.

    Some inputs have vowel sign + consonant in the wrong order;
    NFC/NFKC handles most cases, but we add a safety pass.
    """
    # Re-apply NFC to fix any remaining ordering issues
    text = unicodedata.normalize("NFC", text)
    return text


# ---------------------------------------------------------------------------
# Zero-width character cleanup
# ---------------------------------------------------------------------------

# Zero-width chars to REMOVE (keep ZWNJ U+200C and ZWJ U+200D)
_ZW_REMOVE = re.compile(
    "[\u200B"     # zero-width space
    "\u200E"      # left-to-right mark
    "\u200F"      # right-to-left mark
    "\u202A-\u202E"  # bidi embedding / override
    "\uFEFF"      # byte order mark / ZWNBSP
    "]"
)


def remove_zero_width(text: str) -> str:
    """Remove zero-width formatting characters (preserves ZWNJ and ZWJ)."""
    return _ZW_REMOVE.sub("", text)

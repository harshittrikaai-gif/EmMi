"""
Data preprocessing pipeline for Emmit.

Provides quality filtering, deduplication, normalization, and tokenization
for multilingual text corpora.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any, Dict, Optional, Set

from emmit.tokenizer.config import INDIC_LANGUAGES, LANGUAGE_SCRIPT_MAP
from emmit.tokenizer.normalization import normalize_indic_text


class DataPreprocessor:
    """
    Scalable text preprocessing pipeline.

    Features:
      * Language detection (placeholder — plug in ``fasttext`` or ``langdetect``)
      * Quality filtering (length, script consistency, repetition)
      * MinHash-style deduplication via SHA-256
      * Script-aware normalisation
    """

    def __init__(self, max_seq_len: int = 2048):
        self.max_seq_len = max_seq_len
        self.seen_hashes: Set[str] = set()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_document(self, document: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single text document through the full pipeline.

        Args:
            document: dict with at least ``"text"`` and ``"source"`` keys.

        Returns:
            Processed dict with ``text``, ``language``, ``source`` — or ``None``
            if the document is filtered out.
        """
        text: str = document.get("text", "")
        source: str = document.get("source", "unknown")

        # 1. Language detection
        lang = self.detect_language(text)

        # 2. Quality filter
        if not self.quality_filter(text, lang):
            return None

        # 3. Deduplication
        doc_hash = self._compute_hash(text)
        if doc_hash in self.seen_hashes:
            return None
        self.seen_hashes.add(doc_hash)

        # 4. Normalization
        text = self.normalize_text(text, lang)

        return {"text": text, "language": lang, "source": source}

    # ------------------------------------------------------------------
    # Language detection (stub — replace with fasttext / langdetect)
    # ------------------------------------------------------------------

    @staticmethod
    def detect_language(text: str) -> str:
        """
        Detect the primary language of *text*.

        Returns an ISO 639-1 code.  The default implementation uses a
        simple script-frequency heuristic; for production, plug in
        ``fasttext`` ``lid.176.bin``.
        """
        script_counts: Dict[str, int] = {}
        for ch in text:
            if not ch.isalpha():
                continue
            try:
                name = unicodedata.name(ch, "")
            except ValueError:
                continue
            script = name.split()[0] if name else "UNKNOWN"
            script_counts[script] = script_counts.get(script, 0) + 1

        if not script_counts:
            return "en"

        dominant = max(script_counts, key=script_counts.get)  # type: ignore[arg-type]

        # Map dominant script → language (best guess)
        script_to_lang = {
            "DEVANAGARI": "hi",
            "BENGALI": "bn",
            "TAMIL": "ta",
            "TELUGU": "te",
            "KANNADA": "kn",
            "MALAYALAM": "ml",
            "GUJARATI": "gu",
            "GURMUKHI": "pa",
            "ORIYA": "or",
            "ARABIC": "ar",
            "CJK": "zh",
            "HANGUL": "ko",
            "HIRAGANA": "ja",
            "KATAKANA": "ja",
            "CYRILLIC": "ru",
            "LATIN": "en",
        }
        return script_to_lang.get(dominant, "en")

    # ------------------------------------------------------------------
    # Quality filtering
    # ------------------------------------------------------------------

    def quality_filter(self, text: str, lang: str) -> bool:
        """Return ``True`` if the document passes quality heuristics."""

        # Length bounds
        if len(text) < 100 or len(text) > 1_000_000:
            return False

        # Average word length sanity check
        words = text.split()
        if len(words) == 0:
            return False
        avg_word_len = len(text) / len(words)
        if avg_word_len > 25:
            return False

        # Script consistency for Indic
        if lang in INDIC_LANGUAGES:
            if not self._check_script_consistency(text, lang):
                return False

        # Excessive repetition
        if self._has_excessive_repetition(text):
            return False

        return True

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_text(text: str, lang: str) -> str:
        """Normalize text, dispatching to Indic normalizer when needed."""
        if lang in INDIC_LANGUAGES:
            return normalize_indic_text(text, lang)
        # Default: NFKC
        return unicodedata.normalize("NFKC", text)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _check_script_consistency(text: str, lang: str) -> bool:
        """Ensure ≥ 70 % of alphabetic chars belong to the expected script."""
        expected_script = LANGUAGE_SCRIPT_MAP.get(lang)
        if not expected_script:
            return True

        script_chars = 0
        total_alpha = 0
        for ch in text:
            if not ch.isalpha():
                continue
            total_alpha += 1
            try:
                name = unicodedata.name(ch, "")
            except ValueError:
                continue
            if name.upper().startswith(expected_script.upper()):
                script_chars += 1

        if total_alpha == 0:
            return True
        return (script_chars / total_alpha) >= 0.70

    @staticmethod
    def _has_excessive_repetition(text: str, threshold: float = 0.3) -> bool:
        """Detect if > ``threshold`` fraction of lines are duplicates."""
        lines = text.strip().split("\n")
        if len(lines) < 5:
            return False
        unique = set(lines)
        dup_ratio = 1.0 - len(unique) / len(lines)
        return dup_ratio > threshold

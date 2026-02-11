"""
PyTorch Dataset for Emmit pre-training.

Supports:
  • Loading pre-tokenised data from ``.bin`` / ``.pt`` files
  • Sequence packing (multiple short docs into one training sequence)
  • On-the-fly text tokenisation for quick prototyping
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset


class EmmitDataset(Dataset):
    """
    Dataset that serves fixed-length token sequences for causal LM training.

    Two modes of operation:

    1. **Pre-tokenised** — pass ``token_files`` (list of ``.pt`` files, each
       containing a 1-D ``LongTensor`` of token ids).  Documents are
       concatenated and split into chunks of ``max_seq_len``.

    2. **Raw text** — pass ``text_files`` and a ``tokenizer``.  Texts are
       tokenised on-the-fly, concatenated, and chunked.  Intended for
       quick prototyping on small data.
    """

    def __init__(
        self,
        max_seq_len: int = 2048,
        token_files: Optional[List[str | Path]] = None,
        text_files: Optional[List[str | Path]] = None,
        tokenizer=None,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tokens: torch.Tensor  # 1-D LongTensor

        if token_files:
            self.tokens = self._load_token_files(token_files)
        elif text_files and tokenizer is not None:
            self.tokens = self._tokenize_text_files(text_files, tokenizer)
        else:
            raise ValueError(
                "Provide either `token_files` or (`text_files` + `tokenizer`)."
            )

        # Number of full chunks
        self.num_samples = len(self.tokens) // self.max_seq_len

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx * self.max_seq_len
        end = start + self.max_seq_len

        input_ids = self.tokens[start:end].clone()
        labels = input_ids.clone()

        return {"input_ids": input_ids, "labels": labels}

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_token_files(paths: List[str | Path]) -> torch.Tensor:
        """Concatenate multiple ``.pt`` token files into a single 1-D tensor."""
        all_tokens: List[torch.Tensor] = []
        for p in paths:
            t = torch.load(p, map_location="cpu", weights_only=True)
            if t.dim() > 1:
                t = t.view(-1)
            all_tokens.append(t.long())
        return torch.cat(all_tokens)

    @staticmethod
    def _tokenize_text_files(
        paths: List[str | Path], tokenizer
    ) -> torch.Tensor:
        """
        Read raw text files, tokenise, and concatenate into a 1-D tensor.

        ``tokenizer`` must expose an ``.encode(text) -> List[int]`` method
        (e.g. a SentencePiece processor).
        """
        all_ids: List[int] = []
        for p in paths:
            text = Path(p).read_text(encoding="utf-8")
            ids = tokenizer.encode(text)
            all_ids.extend(ids)
        return torch.tensor(all_ids, dtype=torch.long)

    # ------------------------------------------------------------------
    # Convenience factory
    # ------------------------------------------------------------------

    @classmethod
    def from_directory(
        cls,
        data_dir: str | Path,
        max_seq_len: int = 2048,
        extension: str = ".pt",
    ) -> "EmmitDataset":
        """
        Create a dataset from all files with ``extension`` in ``data_dir``.
        """
        data_dir = Path(data_dir)
        files = sorted(data_dir.glob(f"*{extension}"))
        if not files:
            raise FileNotFoundError(
                f"No {extension} files found in {data_dir}"
            )
        return cls(max_seq_len=max_seq_len, token_files=files)

"""Sentence-transformer embedder adapted for Concordia."""

from __future__ import annotations

import numpy as np


class MpnetEmbedder:
    """Embedder using HuggingFace, adapted to Concordia."""

    def __init__(self, model):
        self._model = model

    def __call__(self, text: str) -> np.ndarray:
        return self._model.encode(text, show_progress_bar=False).astype(np.float32)

    def embed(self, text: str) -> np.ndarray:
        return self._model.encode(text, show_progress_bar=False).astype(np.float32)

    def embed_sentence(self, text: str) -> np.ndarray:
        return self.embed(text)

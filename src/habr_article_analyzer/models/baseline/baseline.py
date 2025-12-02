from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from habr_article_analyzer.models.base import BaseHubClassifier
from habr_article_analyzer.models.encoders.word2vec_encoder import (
    BilingualWord2VecEncoder,
)
from habr_article_analyzer.models.predictors.knn_predictor import KNNPredictor


class BaselineWord2VecKNN(BaseHubClassifier):
    """
    End-to-end model combining:
        - text encoder -> text vector
        - hub encoder -> hub vector
        - KNN predictor -> probability
    Works purely through encoder + predictor interfaces.
    """

    def __init__(
        self,
        text_encoder: BilingualWord2VecEncoder,
        hub_encoder: BilingualWord2VecEncoder,
        predictor: Optional[KNNPredictor] = None,
    ):
        if text_encoder is None or hub_encoder is None:
            raise ValueError("Encoders must be provided")

        self.text_encoder = text_encoder
        self.hub_encoder = hub_encoder
        self.predictor = predictor or KNNPredictor()

    def _encode_pair(self, text: str, hub: str) -> np.ndarray:
        """Compute concatenated embedding of (text, hub)."""
        text_vec = self.text_encoder.encode(text)
        hub_vec = self.hub_encoder.encode(hub)
        return np.concatenate([text_vec, hub_vec])

    # ------------------------------------------------------------------
    # Interface Methods
    # ------------------------------------------------------------------
    def fit(
        self,
        texts: Iterable[str],
        hubs: Iterable[str],
        labels: Iterable[int],
    ) -> None:
        """Fit model using encoded (text, hub) pairs."""
        X = np.vstack([self._encode_pair(t, h) for t, h in zip(texts, hubs)])
        y = np.asarray(labels)
        self.predictor.fit(X, y)

    def predict_proba(self, text: str, hub: str) -> float:
        """Predict probability that text belongs to hub."""
        pair_vec = self._encode_pair(text, hub)
        return float(self.predictor.predict_proba(pair_vec))

    # ------------------------------------------------------------------
    # Serialization (pickle)
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Save entire model (encoders + predictor)."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> Any:
        """Load model from a pickle file."""
        with open(path, "rb") as f:
            return pickle.load(f)

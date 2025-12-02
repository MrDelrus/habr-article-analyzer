import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from habr_article_analyzer.models.base import BaseHubClassifier
from habr_article_analyzer.models.encoders.base import TextEncoder
from habr_article_analyzer.models.predictors.base import TrainablePredictor


class ModelWrapper(BaseHubClassifier):
    """
    Generic wrapper for any encoder + predictor.
    Converts (text, hub) to vector using encoder before sending to predictor.
    """

    def __init__(self, encoder: TextEncoder, predictor: TrainablePredictor):
        self.encoder = encoder
        self.predictor = predictor

    def _encode_pair(self, text: str, hub: str) -> np.ndarray:
        """Concatenate text and hub embeddings."""
        text_vec = self.encoder.encode(text)
        hub_vec = self.encoder.encode(hub)
        return np.concatenate([text_vec, hub_vec])

    def fit(
        self,
        texts: Iterable[str],
        hubs: Iterable[str],
        labels: Iterable[int],
    ) -> None:
        """Encode texts+hubs to vectors and train predictor on them."""
        X = np.vstack([self._encode_pair(t, h) for t, h in zip(texts, hubs)])
        y = np.asarray(labels)
        self.predictor.fit(X, y)

    def predict(self, text: str, hub: str) -> int:
        vec = self._encode_pair(text, hub)
        return self.predictor.predict(vec)

    def predict_proba(self, text: str, hub: str) -> float:
        vec = self._encode_pair(text, hub)
        return float(self.predictor.predict_proba(vec))

    # ---------------------
    # SERIALIZATION
    # ---------------------
    def save(self, path: str | Path) -> None:
        """Save the full wrapper (encoders + predictor)."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> Any:
        """Load the full wrapper from a pickle file."""
        with open(path, "rb") as f:
            return pickle.load(f)

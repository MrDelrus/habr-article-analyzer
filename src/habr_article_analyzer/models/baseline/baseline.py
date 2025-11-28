from typing import Dict, List, Optional

import numpy as np

from habr_article_analyzer.models.base import BaseHubClassifier
from habr_article_analyzer.models.encoders.word2vec_encoder import (
    BilingualWord2VecEncoder,
)
from habr_article_analyzer.models.predictors.knn_predictor import KNNPredictor


class BaselineWord2VecKNN(BaseHubClassifier):
    """
    End-to-end model:
        text_encoder -> text vector
        hub_encoder -> hub vector
        predictor -> probability
    """

    def __init__(
        self,
        text_encoder: BilingualWord2VecEncoder,
        hub_encoder: BilingualWord2VecEncoder,
        predictor: Optional[KNNPredictor] = None,
    ):
        if text_encoder is None or hub_encoder is None:
            raise ValueError(
                "BilingualWord2VecEncoder instances must be provided for the encoders"
            )
        self.text_encoder = text_encoder
        self.hub_encoder = hub_encoder
        self.predictor = predictor or KNNPredictor()

    def _encode_pair(self, text: str, hub: str) -> np.ndarray:
        """Private helper: concatenate embeddings of text and hub."""
        text_vec = self.text_encoder.encode(text)
        hub_vec = self.hub_encoder.encode(hub)
        return np.concatenate([text_vec, hub_vec])

    # ------------------------
    # Interface methods
    # ------------------------
    def predict_proba(self, text: str, hub: str) -> float:
        return self.predictor.predict_proba(self._encode_pair(text, hub))

    def fit(self, texts: List[str], hubs: List[str], labels: List[int]) -> None:
        X = np.vstack([self._encode_pair(t, h) for t, h in zip(texts, hubs)])
        y = np.array(labels)
        self.predictor.fit(X, y)

    def save(self, paths: Dict[str, str]) -> None:
        self.text_encoder.save(paths["text_encoder"])
        self.hub_encoder.save(paths["hub_encoder"])
        self.predictor.save(paths["predictor"])

    def load(self, paths: Dict[str, str]) -> None:
        self.text_encoder.load(paths["text_encoder"])
        self.hub_encoder.load(paths["hub_encoder"])
        self.predictor.load(paths["predictor"])

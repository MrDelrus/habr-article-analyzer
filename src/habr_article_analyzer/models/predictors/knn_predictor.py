import os

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from habr_article_analyzer.models.predictors.base import TrainablePredictor


class KNNPredictor(TrainablePredictor):
    """
    Simple KNN binary classifier for baseline probability estimation.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "distance",
        probability_threshold: float = 0.5,
    ):
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
        )
        self.probability_threshold = probability_threshold

    # --------------------
    # Training
    # --------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    # --------------------
    # Inference
    # --------------------
    def predict_proba(self, x: np.ndarray) -> float:
        proba = self.model.predict_proba(x.reshape(1, -1))[0, 1]
        return float(proba)

    def predict(self, x: np.ndarray) -> int:
        return int(self.predict_proba(x) >= self.probability_threshold)

    # --------------------
    # Loading & saving
    # --------------------
    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self.model = joblib.load(path)

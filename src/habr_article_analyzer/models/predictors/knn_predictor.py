import os

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class KNNPredictor:
    """
    Simple KNN binary classifier for baseline probability estimation.
    """

    def __init__(self, n_neighbors: int = 5, weights: str = "distance"):
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
        )

    # --------------------
    # Training
    # --------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    # --------------------
    # Inference
    # --------------------
    def predict_proba(self, x: np.ndarray) -> float:
        """
        x: vector (n+m,)
        Returns probability of class 1.
        """
        proba = self.model.predict_proba(x.reshape(1, -1))[0, 1]
        return float(proba)

    # --------------------
    # Loading & saving
    # --------------------
    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self.model = joblib.load(path)

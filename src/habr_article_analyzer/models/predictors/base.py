from abc import ABC, abstractmethod

import numpy as np


class TrainablePredictor(ABC):
    """Interface for vector-based binary classifiers."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model on vectorized samples.

        X: (N, D) matrix of concatenated text+hub embeddings
        y: (N,) binary labels
        """
        pass

    @abstractmethod
    def predict_proba(self, x: np.ndarray) -> float:
        """
        Predict probability of class 1 for a single vector x.
        x: (D,) vector
        """
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> int:
        """
        Predict binary label for a single vector x.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Serialize model to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk."""
        pass

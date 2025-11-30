from abc import ABC, abstractmethod
from typing import Iterable


class TrainablePredictor(ABC):
    """Interface for binary prediction models."""

    @abstractmethod
    def fit(
        self,
        texts: Iterable[str],
        hubs: Iterable[str],
        labels: Iterable[int],
    ) -> None:
        """Train the model."""
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, text: str, hub: str) -> float:
        """Returns probability that the given text belongs to the hub."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, text: str, hub: str) -> int:
        """Returns class label (0 or 1)."""
        raise NotImplementedError

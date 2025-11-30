from abc import ABC, abstractmethod


class Predictor(ABC):
    """Interface for binary prediction models."""

    @abstractmethod
    def predict_proba(self, text: str, hub: str) -> float:
        """
        Returns probability that the given text belongs to the hub.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, text: str, hub: str) -> int:
        """
        Returns class label (0 or 1).
        """
        raise NotImplementedError

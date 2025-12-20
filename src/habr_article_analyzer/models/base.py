from abc import ABC, abstractmethod
from pathlib import Path


class BaseHubClassifier(ABC):
    """
    Base interface for any model that predicts the probability
    of a text belonging to a hub.
    """

    @abstractmethod
    def predict_proba(self, text: str, hub: str) -> float:
        """
        Return probability that `text` belongs to `hub`.
        """
        pass

    @abstractmethod
    def fit(self, texts: list[str], hubs: list[str], labels: list[int]) -> None:
        """
        Train / fit the model on a dataset.
        """
        pass

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """
        Save the model weights and components to disk.
        """
        pass

    @abstractmethod
    def load(self, paths: str | Path) -> None:
        """
        Load the model weights and components from disk.
        """
        pass

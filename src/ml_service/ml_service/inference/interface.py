from abc import ABC, abstractmethod


class BaseHubClassifierInference(ABC):
    @abstractmethod
    def predict_proba(self, text: str, hub: str) -> float:
        """Return probability that `text` belongs to `hub`."""
        pass

    @abstractmethod
    def predict(self, text: str, hub: str) -> int:
        """Return 1 if that `text` belongs to `hub` and 0 else."""
        pass

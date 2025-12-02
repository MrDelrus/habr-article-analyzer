import pickle
from pathlib import Path
from typing import Any, Iterable

from habr_article_analyzer.models.base import BaseHubClassifier
from habr_article_analyzer.models.encoders.base import TextEncoder
from habr_article_analyzer.models.predictors.base import TrainablePredictor


class ModelWrapper(BaseHubClassifier):
    """
    Combines encoder + predictor into a single end-to-end model.
    Works fully through abstract interfaces.
    """

    def __init__(self, encoder: TextEncoder, predictor: TrainablePredictor):
        self.encoder = encoder
        self.predictor = predictor

    def fit(
        self,
        texts: Iterable[str],
        hubs: Iterable[str],
        labels: Iterable[int],
    ) -> None:
        self.predictor.fit(texts, hubs, labels)

    def predict(self, text: str, hub: str) -> int:
        return self.predictor.predict(text, hub)

    def predict_proba(self, text: str, hub: str) -> float:
        return self.predictor.predict_proba(text, hub)

    # ---------------------
    # SERIALIZATION
    # ---------------------

    def save(self, path: str | Path) -> None:
        """
        Save the whole wrapper including encoder and predictor.
        Works as long as both are pickle-serializable.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> Any:
        with open(path, "rb") as f:
            return pickle.load(f)

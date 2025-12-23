import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

url_stopwords_ru = (
    "https://raw.githubusercontent.com/stopwords-iso/"
    + "stopwords-ru/master/stopwords-ru.txt"
)


def get_text(url: str, encoding: str = "utf-8", to_lower: bool = True) -> Any:
    url = str(url)
    if url.startswith("http"):
        r = requests.get(url)
        if not r.ok:
            r.raise_for_status()
        return r.text.lower() if to_lower else r.text
    elif os.path.exists(url):
        with open(url, encoding=encoding) as f:
            return f.read().lower() if to_lower else f.read()
    else:
        raise Exception("parameter [url] can be either URL or a filename")


russian_stopwords = get_text(url_stopwords_ru).splitlines()
bilingual_stopwords = set(russian_stopwords) | ENGLISH_STOP_WORDS
bilingual_stopwords = list(bilingual_stopwords)


class TextEncoder:
    max_features: int
    vectorizer: TfidfVectorizer

    def __init__(self, max_features: int = 5000):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=10,
            max_df=0.9,
            lowercase=True,
            stop_words=bilingual_stopwords,
            token_pattern=r"(?u)\b[a-zа-яёA-ZА-ЯЁ]{3,}\b",
        )

    def fit(self, texts: list[str]) -> None:
        self.vectorizer.fit(texts)

    def transform(self, texts: list[str]) -> np.ndarray:
        return np.array(self.vectorizer.transform(texts).todense())

    def state_dict(self) -> dict[str, Any]:
        vectorizer_bytes = pickle.dumps(self.vectorizer)
        return {"max_features": self.max_features, "vectorizer_bytes": vectorizer_bytes}

    @classmethod
    def load_state_dict(cls, state_dict: dict[str, Any]) -> Any:
        encoder = cls(max_features=state_dict["max_features"])
        encoder.max_features = state_dict["max_features"]
        encoder.vectorizer = pickle.loads(state_dict["vectorizer_bytes"])
        return encoder

    def save(self, path: Path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: Path) -> Any:
        state_dict = torch.load(path, weights_only=True)        
        return cls.load_state_dict(state_dict)



class A:
    def __init__(self, x):
        self.x = x


from pathlib import Path
from typing import Any
import numpy as np
import torch
from ml_training.models.encoders.hub_averaging_encoder import HubEncoder
from ml_training.models.encoders.tf_idf_encoder import TextEncoder
from ml_training.models.predictors.ranking_nn import RankingModel
from ml_training.models.wrapper import ModelWrapper


class BoWDSSM(ModelWrapper):
    """
    Bag of Words encoders for text and hub + some neural layers as predictor.    
    """
    def __init__(
        self, 
        text_encoder: TextEncoder,
        hub_encoder: HubEncoder,
        predictor: RankingModel
    ):
        self.text_encoder = text_encoder
        self.hub_encoder = hub_encoder
        self.predictor = predictor
    
    def _encode_pair(self, text: str, hub: str) -> np.ndarray:
        text_vec = self.text_encoder.transform(text)
        hub_vec = self.hub_encoder.transform(hub)
        return np.concatenate([text_vec, hub_vec])

    def state_dict(self) -> Any:
        return {
            "text_encoder": self.text_encoder.state_dict(),
            "hub_encoder": self.hub_encoder.state_dict(),
            "predictor": self.predictor.state_dict() 
        }

    @classmethod
    def load_state_dict(cls, state_dict: dict[str, Any]) -> Any:
        text_encoder = TextEncoder().load_state_dict(state_dict["text_encoder"])
        hub_encoder = HubEncoder().load_state_dict(state_dict["hub_encoder"])
        predictor = RankingModel().load_state_dict(state_dict["predictor"])
        return BoWDSSM(
            text_encoder,
            hub_encoder,
            predictor
        )

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> Any:
        state_dict = torch.load(path, weights_only=True)
        return cls.load_state_dict(state_dict)

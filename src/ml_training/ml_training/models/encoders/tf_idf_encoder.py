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

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.max_features = state_dict["max_features"]
        self.vectorizer = pickle.loads(state_dict["vectorizer_bytes"])

    def save(self, path: Path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: Path) -> Any:
        state_dict = torch.load(path, weights_only=True)
        encoder = cls(max_features=state_dict["max_features"])
        encoder.load_state_dict(state_dict)
        return encoder

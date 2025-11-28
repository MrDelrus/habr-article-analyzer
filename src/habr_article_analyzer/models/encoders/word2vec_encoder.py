import re
from typing import Iterable, List

import numpy as np
from gensim.models import KeyedVectors


class BilingualWord2VecEncoder:
    """
    Encoder for mixed Russian + English texts using two Word2Vec/fastText models.
    """

    def __init__(
        self,
        kv_ru: KeyedVectors,
        kv_en: KeyedVectors,
        aggregation: str = "mean",
        lowercase: bool = True,
    ):
        """
        kv_ru: KeyedVectors for Russian
        kv_en: KeyedVectors for English
        aggregation: "mean", "sum", "max"
        """
        assert aggregation in {"mean", "sum", "max"}
        self.kv_ru = kv_ru
        self.kv_en = kv_en
        self.aggregation = aggregation
        self.lowercase = lowercase
        assert kv_ru.vector_size == kv_en.vector_size
        self.dim = kv_ru.vector_size

    # ---------------------------
    # Tokenization / language check
    # ---------------------------
    @staticmethod
    def tokenize(text: str) -> List[str]:
        # Simple whitespace + punctuation split
        tokens = []
        for tok in text.split():
            tok = tok.strip(".,!?;:\"'()[]{}<>—-")
            if tok:
                tokens.append(tok)
        return tokens

    @staticmethod
    def is_english(token: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z0-9]+", token))

    @staticmethod
    def is_russian(token: str) -> bool:
        return bool(re.fullmatch(r"[А-Яа-яёЁ0-9]+", token))

    # ---------------------------
    # Encode
    # ---------------------------
    def encode(self, text: str) -> np.ndarray:
        tokens = self.tokenize(text)
        if self.lowercase:
            tokens = [t.lower() for t in tokens]

        vecs = []
        for t in tokens:
            if self.is_russian(t) and t in self.kv_ru:
                vecs.append(self.kv_ru[t])
            elif self.is_english(t) and t in self.kv_en:
                vecs.append(self.kv_en[t])

        if not vecs:
            # all OOV → zero vector
            return np.zeros(self.dim, dtype=float)

        arr = np.vstack(vecs)

        if self.aggregation == "mean":
            return np.mean(arr, axis=0)
        elif self.aggregation == "sum":
            return np.sum(arr, axis=0)
        else:  # max
            return np.max(arr, axis=0)

    def encode_batch(self, texts: Iterable[str]) -> np.ndarray:
        return np.vstack([self.encode(t) for t in texts])

from collections import abc
from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

Index = type[str | int | list[str] | list[int] | np.ndarray[int] | None]


class Target:
    dataset: pd.DataFrame
    targets: pd.Series
    column_name: str
    labels: list[str]
    binary_mask: np.ndarray | sp.spmatrix
    sparse: bool

    def __init__(self, dataset: pd.DataFrame, column_name: str, sparse: bool = False):
        self.dataset = dataset
        self.column_name = column_name
        self.targets = Target._as_str_list(dataset[column_name])
        self.labels = Target._get_labels(self.targets)
        self.sparse = sparse
        self.binary_mask = MultiLabelBinarizer(
            classes=self.labels,
            sparse_output=sparse,
        ).fit_transform(self.targets)

    @staticmethod
    def _get_labels(targets: pd.Series) -> list[str]:
        labels = set()
        for labels_item in targets:
            for label_item in labels_item:
                labels.add(label_item)
        return sorted(list(labels))

    @staticmethod
    def _as_str_list(targets: pd.Series) -> pd.Series:
        return targets.apply(lambda x: [x] if isinstance(x, str) else x)

    def label_to_id(self, label: str) -> int:
        if label not in self.labels:
            raise KeyError("{} not in labels".format(label))
        return self.labels.index(label)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: Index) -> np.ndarray | sp.spmatrix:
        """Возвращает подмножество binary_mask по индексу."""
        if isinstance(idx, str):
            idx = self.label_to_id(idx)
        elif isinstance(idx, abc.Iterable) and not isinstance(idx, np.ndarray):
            idx = np.array(
                list(
                    map(lambda x: self.label_to_id(x) if isinstance(x, str) else x, idx)
                )
            )

        if sp.issparse(self.binary_mask):
            result = self.binary_mask[:, idx]
            if isinstance(idx, (int, np.integer)):
                return result.toarray().ravel()
            return result
        else:
            return self.binary_mask[:, idx]

    def get_coverage(self, idx: Index = None) -> np.ndarray | float:
        """Вычисляет покрытие (долю документов с хотя бы одним из указанных лейблов)."""
        if idx is None:
            idx = self.labels

        binary_submask = self[idx]

        if sp.issparse(binary_submask):
            if binary_submask.shape[1] == 1:
                return binary_submask.getnnz() / binary_submask.shape[0]
            else:
                has_any_label = binary_submask.max(axis=1).toarray().ravel() > 0
                return has_any_label.mean()
        else:
            if len(binary_submask.shape) == 1:
                return binary_submask.mean()
            return binary_submask.any(axis=1).mean()

    def get_sizes(self, idx: Index = None) -> np.ndarray | int:
        """Возвращает количество документов для каждого лейбла."""
        if idx is None:
            idx = self.labels

        binary_submask = self[idx]

        if sp.issparse(binary_submask):
            result = np.array(binary_submask.sum(axis=0)).ravel()
            if isinstance(idx, (str, int, np.integer)):
                return int(result[0]) if result.size == 1 else int(result)
            return result
        else:
            result = binary_submask.sum(axis=0)
            if np.isscalar(result) or (hasattr(result, "size") and result.size == 1):
                return int(result)
            return result.astype(int)

    def get_top_words_per_label(
        self,
        text_column: str,
        n_words: int = 10,
        method: Literal["frequency", "tfidf"] = "tfidf",
        max_features: int = 10000,
        min_df: int = 2,
        ngram_range: tuple[int, int] = (1, 1),
    ) -> dict[str, list[tuple[str, float]]]:
        """Извлекает топ-N слов для каждого лейбла."""

        if text_column not in self.dataset.columns:
            raise ValueError(f"Column {text_column} not found in dataset")

        labels_to_process = self.labels
        texts = self.dataset[text_column].fillna("")

        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            ngram_range=ngram_range,
            dtype=np.float32,
        )

        text_matrix = vectorizer.fit_transform(texts)
        feature_names = np.array(vectorizer.get_feature_names_out())

        if sp.issparse(self.binary_mask):
            label_matrix = self.binary_mask.astype(np.float32)
        else:
            label_matrix = sp.csr_matrix(self.binary_mask, dtype=np.float32)

        results = {}

        if method == "frequency":
            label_word_matrix = label_matrix.T @ text_matrix

            for i, label in enumerate(labels_to_process):
                if sp.issparse(label_word_matrix):
                    scores = label_word_matrix[i].toarray().ravel()
                else:
                    scores = np.array(label_word_matrix[i]).ravel()

                if len(scores) > n_words:
                    top_indices = np.argpartition(scores, -n_words)[-n_words:]
                else:
                    top_indices = np.arange(len(scores))
                top_indices = top_indices[np.argsort(-scores[top_indices])]

                results[label] = [
                    (feature_names[idx], float(scores[idx]))
                    for idx in top_indices
                    if scores[idx] > 0
                ]

        elif method == "tfidf":
            label_doc_counts = np.array(label_matrix.sum(axis=0)).ravel()

            label_word_matrix = label_matrix.T @ text_matrix

            total_word_freq = np.array(text_matrix.sum(axis=0)).ravel() + 1e-10

            for i, label in enumerate(labels_to_process):
                if label_doc_counts[i] == 0:
                    results[label] = []
                    continue

                if sp.issparse(label_word_matrix):
                    label_word_scores = label_word_matrix[i].toarray().ravel()
                else:
                    label_word_scores = np.array(label_word_matrix[i]).ravel()

                avg_tfidf = label_word_scores / max(label_doc_counts[i], 1)

                scores = avg_tfidf / (total_word_freq / len(texts))

                if len(scores) > n_words:
                    top_indices = np.argpartition(scores, -n_words)[-n_words:]
                else:
                    top_indices = np.arange(len(scores))
                top_indices = top_indices[np.argsort(-scores[top_indices])]

                results[label] = [
                    (feature_names[idx], float(scores[idx]))
                    for idx in top_indices
                    if scores[idx] > 0
                ]
        else:
            raise ValueError(
                f"Method {method} not supported. Use 'frequency' or 'tfidf'"
            )

        return results

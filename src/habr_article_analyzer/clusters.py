from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import pairwise_distances

from habr_article_analyzer.targets import Target


class Embedder:
    def __init__(self) -> None:
        pass

    def encode(self, texts: list[str]) -> np.ndarray:
        pass


class DatasetEmbedder(Embedder):
    target: Target

    def __init__(self, target: Target):
        self.target = target

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.target[texts].T


class JaccardLabelsEmbedder(Embedder):
    target: Target

    def __init__(self, target: Target):
        self.target = target

    def encode(self, texts: list[str]) -> np.ndarray:
        target_mask = self.target[texts]
        jaccard_dist = pairwise_distances(target_mask.T, metric="jaccard")
        jaccard_sim = jaccard_dist
        return jaccard_sim


Clusterization = BaseEstimator


class TextClusterization:
    embedder: Embedder
    clusters: Clusterization

    def __init__(self, embedder: Embedder, clusters: Clusterization):
        self.embedder = embedder
        self.clusters = clusters

    def fit_predict(self, X: list[str], **kwargs) -> np.ndarray:
        return self.clusters.fit_predict(self.embedder.encode(X), **kwargs)

    def get_clusters(self, X: list[str], **kwargs) -> dict[int, list[str]]:
        return TextClusterization._get_clusters(self.fit_predict(X, **kwargs), X)

    @staticmethod
    def _get_clusters(clusters: np.ndarray, X: list[str]) -> dict[int, list[str]]:
        result = defaultdict(list)
        for key, value in zip(clusters, X):
            result[int(key)].append(value)
        return result

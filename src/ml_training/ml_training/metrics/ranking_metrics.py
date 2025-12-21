from typing import Callable

import numpy as np

from ml_training.data.targets import Target


class RankingMetrics:
    def __init__(self, target: Target, predicts: np.ndarray):
        assert (
            len(predicts.shape) == 2
        ), "predicts must be formatted as a 2d array: [n_texts x n_labels]"
        assert (
            target.binary_mask.shape == predicts.shape
        ), "target inner representation should match with predicts shape"
        self.target = target
        self.predicts = predicts
        self.sorted_predicts = predicts.argsort(axis=1)[:, ::-1]

    def dcg(
        self, discount: Callable[[int], float] = lambda rank: 1.0 / (rank + 1.0)
    ) -> np.ndarray:
        discount_array = np.array([discount(rank) for rank in range(len(self.target))])
        row_dcgs = []
        for i, predicts in enumerate(self.sorted_predicts):
            scores = self.target[predicts][i].toarray().ravel()
            row_dcgs.append(scores.T @ discount_array)

        return np.array(row_dcgs)

    def ndcg(
        self, discount: Callable[[int], float] = lambda rank: 1.0 / (rank + 1.0)
    ) -> np.ndarray:
        counts = np.array(self.target.binary_mask.sum(axis=1).flatten())
        discount_array = np.array([discount(rank) for rank in range(len(self.target))])
        norm_dcgs = []
        for count in counts[0]:
            norm_dcgs.append(np.sum(discount_array[:count]))

        return self.dcg(discount) / np.array(norm_dcgs)

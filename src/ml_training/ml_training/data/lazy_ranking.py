from typing import Literal, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ml_training.data.targets import Target


class LazyRankingDatasetBase(Dataset):
    def __init__(
        self, text_embeddings: np.ndarray, hub_embeddings: np.ndarray, target: Target
    ):
        super().__init__()
        self.text_embeddings = text_embeddings
        self.hub_embeddings = hub_embeddings
        self.target = target
        self.id_mapping = self._id_mapping()

    def _id_mapping(self) -> np.ndarray:
        pass

    def __getitem__(self, index: int | np.ndarray) -> Tuple[torch.Tensor, int]:
        text_hub_id = self.id_mapping[index]
        return (
            torch.from_numpy(
                np.concatenate(
                    [
                        self.text_embeddings[text_hub_id[0]],
                        self.hub_embeddings[text_hub_id[1]],
                    ],
                    dtype=np.float32,
                ),
            ),
            torch.tensor(
                self.target[text_hub_id[1]][text_hub_id[0]], dtype=torch.float32
            ),
        )

    def __len__(self) -> int:
        return len(self.id_mapping)


class FullLazyRankingDataset(LazyRankingDatasetBase):
    class LazyMapping:
        def __init__(self, n_docs: int, n_hubs: int):
            self.n_docs = n_docs
            self.n_hubs = n_hubs

        def __getitem__(self, index: int | np.ndarray) -> np.ndarray:
            return np.array([index // self.n_hubs, index % self.n_hubs]).T

        def __len__(self) -> int:
            return self.n_docs * self.n_hubs

    def __init__(
        self, text_embeddings: np.ndarray, hub_embeddings: np.ndarray, target: Target
    ):
        super().__init__(text_embeddings, hub_embeddings, target)

    def _id_mapping(self) -> LazyMapping:
        return FullLazyRankingDataset.LazyMapping(
            len(self.text_embeddings), len(self.hub_embeddings)
        )


class SamplingLazyRankingDataset(LazyRankingDatasetBase):
    def __init__(
        self,
        text_embeddings: np.ndarray,
        hub_embeddings: np.ndarray,
        target: Target,
        positives_cnt: int,
        negatives_cnt: int,
        sampling_strategy: Literal["fixed_cnt", "label_proportional"] = "fixed_cnt",
    ):
        self.positives_cnt = positives_cnt
        self.negatives_cnt = negatives_cnt
        self.sampling_strategy = sampling_strategy
        super().__init__(text_embeddings, hub_embeddings, target)

    def _id_mapping(self) -> np.ndarray:
        n_docs, n_labels = self.target.binary_mask.shape
        matrix = self.target.binary_mask.tocsr()

        neg_sampling_probs = None
        if self.sampling_strategy == "label_proportional":
            label_popularity = (
                self.target.get_sizes() + 1e-8
            )  # epsilon to handle zeroes
            # (pretty sure there should be none zeroes, but anyway)
            neg_sampling_probs = (
                label_popularity / label_popularity.sum()  # type: ignore
            )

        results = []
        if self.sampling_strategy in ["fixed_cnt", "label_proportional"]:
            for doc_id in range(n_docs):
                pos_labels = matrix[doc_id].indices  # all 1
                neg_labels = np.setdiff1d(np.arange(n_labels), pos_labels)  # all 0

                sampled_pos_labels = pos_labels
                if len(pos_labels) > self.positives_cnt:
                    sampled_pos_labels = np.random.choice(
                        pos_labels, size=self.positives_cnt, replace=False
                    )

                available_neg_probs = None
                if self.sampling_strategy == "label_proportional":
                    available_neg_probs = neg_sampling_probs[neg_labels]  # type: ignore
                    available_neg_probs = (
                        available_neg_probs / available_neg_probs.sum()
                    )

                sampled_neg_labels = neg_labels
                if len(neg_labels) > self.negatives_cnt:
                    sampled_neg_labels = np.random.choice(
                        neg_labels,
                        size=self.negatives_cnt,
                        replace=False,
                        p=available_neg_probs,
                    )

                for label_id in sampled_pos_labels:
                    results.append([doc_id, label_id])

                for label_id in sampled_neg_labels:
                    results.append([doc_id, label_id])
        else:
            raise ValueError(
                f"Method sampling_strategy {self.sampling_strategy} not supported. \
                  Use 'fixed_cnt' or 'label_proportional'"
            )

        return np.array(results)

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml_training.ml_training.data.targets import Target


class HubEncoder:
    hub_to_vec: dict[str, np.ndarray]
    dim: int

    def __init__(self, dim: int = 5000):
        self.hub_to_vec = {}
        self.dim = dim

    def fit(self, target: Target, text_embeddings: np.ndarray) -> None:
        assert self.dim == text_embeddings.shape[1], "{} != {}".format(
            self.dim, text_embeddings.shape[1]
        )

        for hub in target.labels:
            hub_mask = np.array(target[hub])
            sum_embeds = text_embeddings.T @ hub_mask
            n_matches = sum(hub_mask)
            if n_matches > 0:
                self.hub_to_vec[hub] = (sum_embeds * 1.0) / (n_matches * 1.0)
            else:
                self.hub_to_vec[hub] = sum_embeds

    def transform(self, labels: list[str]) -> np.ndarray:
        result = []

        for hub in labels:
            if hub in self.hub_to_vec.keys():
                result.append(self.hub_to_vec[hub])
            else:
                result.append(np.zeros(self.dim))

        return np.array(result)

    def state_dict(self) -> Any:
        hub_to_vec_tensors = {
            hub: torch.from_numpy(vec) for hub, vec in self.hub_to_vec.items()
        }

        return {"dim": self.dim, "hub_to_vec": hub_to_vec_tensors}

    @classmethod
    def load_state_dict(cls, state_dict: Any) -> Any:
        encoder = cls(dim=state_dict["dim"])
        encoder.hub_to_vec = {
            hub: vec.numpy() for hub, vec in state_dict["hub_to_vec"].items()
        }
        return encoder

    def save(self, path: Path | str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: Path | str) -> Any:
        state_dict = torch.load(path, weights_only=True)
        return cls.load_state_dict(state_dict)

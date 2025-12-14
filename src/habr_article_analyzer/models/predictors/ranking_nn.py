import numpy as np
import torch.nn as nn


class RankingModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [512, 256, 128],
        dropout: float = 0.1,
    ):
        super().__init__()

        prev_dim = input_dim
        layers = []

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.model(x).squeeze(-1)

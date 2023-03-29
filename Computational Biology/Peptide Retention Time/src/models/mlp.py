import torch
from torch import nn


class MLP(nn.Module):

    name = "MLP"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        output_dim: int,
        dropout_prob: float,
        **kwargs
    ):
        super().__init__()

        hidden_layers = hidden_layers * [
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        ]

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            *hidden_layers,
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """MLP forward pass

        Args:
            x (torch.tensor): input tensor

        Returns:
            torch.tensor: output tensor
        """

        return self.layers(x)

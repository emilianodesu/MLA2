"""Multi-Layer Perceptron (MLP) architectures used in experiments.

Provides a flexible baseline MLP with:
- Configurable depth (number of hidden layers)
- Either uniform width (same units each hidden layer) or an explicit list of widths
- BatchNorm + ReLU + Dropout after every hidden Linear layer

The final layer outputs **raw logits** (no softmax). Use ``torch.nn.CrossEntropyLoss`` during
training (which internally applies ``log_softmax`` + NLL). For probabilities at inference, wrap:

    probs = torch.softmax(logits, dim=1)

Example Usage
-------------
Uniform width:
    model = MLPBaseline(input_dim=4096, num_classes=15, hidden_layers=3, hidden_units=256, dropout=0.5)

Variable width:
    model = MLPBaseline(input_dim=4096, num_classes=15, hidden_units=[512, 256, 128], dropout=0.3)
"""
from typing import Union, List
import torch
from torch import nn


class MLPBaseline(nn.Module):
    """Baseline MLP with configurable depth and width.

    Args:
        input_dim: Number of flattened input features (e.g., 64*64=4096 for 64×64 grayscale when flattened).
        num_classes: Number of output classes (logit dimension).
        hidden_layers: Number of hidden layers (ignored if ``hidden_units`` is a list).
        hidden_units: Either an ``int`` (uniform width for each hidden layer) or a list specifying
            the width of each hidden layer in order.
        dropout: Dropout probability applied after each hidden layer (before the final output layer).

    Notes:
        - Each hidden block is: Linear -> BatchNorm1d -> ReLU -> Dropout
        - Output layer is Linear only (no activation); logits returned directly.
        - Provide a list for ``hidden_units`` if you need tapering (e.g., ``[512, 256, 128]``).
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_layers: int = 2,
        hidden_units: Union[int, List[int]] = 256,
        dropout: float = 0.5
    ):
        super().__init__()

        layers = []
        in_features = input_dim

        # If list is provided, use that; otherwise repeat the same size 'hidden_layers' times
        if isinstance(hidden_units, list):
            layer_sizes = hidden_units
        else:
            layer_sizes = [hidden_units] * hidden_layers

        # Hidden layers
        for out_features in layer_sizes:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_features = out_features

        # Output layer
        layers.append(nn.Linear(in_features, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Float tensor of shape ``(batch_size, input_dim)`` (already flattened input features).

        Returns:
            Tensor: Logits of shape ``(batch_size, num_classes)``.
        """
        return self.model(x)

import torch.nn as nn

class LogisticRegressionModel(nn.Module):
    """
    Baseline 0: Logistic Regression for Chinese MNIST.
    A simple linear classifier mapping input_dim -> num_classes.
    """
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, input_dim)
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        return self.linear(x)

class MLPBaseline(nn.Module):
    """
    Baseline 1: Multi-Layer Perceptron (MLP).
    Architecture: input_dim -> 512 -> 128 -> num_classes
    with ReLU, BatchNorm, and Dropout.
    """
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, input_dim)
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        return self.model(x)

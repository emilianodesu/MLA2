"""
Multi-Layer Perceptron (MLP).
"""
import torch
from torch import nn
from torchinfo import summary
from preprocessing_utils import prepare_images, get_dataloaders
from models_utils import train_model, test_model, plot_confusion_matrix, plot_training_process


class MLPBaseline(nn.Module):
    """
    Baseline 1: Multi-Layer Perceptron (MLP).
    Architecture: input_dim -> 512 -> 128 -> num_classes
    with ReLU, BatchNorm, and Dropout.
    """
    def __init__(self, input_dim, num_classes, dropout=0.5):
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

if __name__ == "__main__":
    print(torch.__version__)

    root = prepare_images()
    print(f"Dataset ready at: {root}")

    dataloaders, datasets = get_dataloaders(
        data_dir="data", batch_size=64, image_size=(64, 64), augment=True)

    labels = datasets['train'].classes

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model parameters
    INPUT_DIM = 64 * 64
    NUM_CLASSES = 15
    EPOCHS_MLP = 50
    PATIENCE = 5
    DROPOUT_RATE = 0.5

    # Model definition
    model = MLPBaseline(input_dim=INPUT_DIM, num_classes=NUM_CLASSES, dropout=DROPOUT_RATE).to(device)
    summary(model, (1, INPUT_DIM))  # shape for single sample

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Train
    (
        total_acc_train_plot,
        total_loss_train_plot,
        total_acc_validation_plot,
        total_loss_validation_plot,
    ) = train_model(
        model,
        dataloaders,
        optimizer,
        criterion,
        epochs=EPOCHS_MLP,
        patience=PATIENCE,
        save_path="checkpoints/mlp_model_2.pth",
    )

    plot_training_process(
        total_acc_train_plot,
        total_loss_train_plot,
        total_acc_validation_plot,
        total_loss_validation_plot,
    )

    # Test
    print("\nEvaluating best model on test set...")
    best_model = MLPBaseline(input_dim=INPUT_DIM, num_classes=NUM_CLASSES, dropout=DROPOUT_RATE).to(device)
    best_model.load_state_dict(torch.load("checkpoints/mlp_model_2.pth", map_location=device))

    acc, f1, cm = test_model(best_model, dataloaders["test"], criterion=criterion)

    print("\nFinal Test Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    plot_confusion_matrix(cm, labels)

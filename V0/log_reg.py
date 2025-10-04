"""
Logistic Regression
"""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from V0.models_utils import train_one_epoch, evaluate, save_model, test
from preprocessing_utils import get_dataloaders


class LogisticRegressionModel(nn.Module):
    """
    Baseline 0: Logistic Regression for Chinese MNIST.
    A simple linear classifier mapping input_dim -> num_classes.
    """

    def __init__(self, input_dim, num_classes):
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


def train_lr(input_dim,
             num_classes,
             optimizer_type="sgd",
             batch_size=64,
             epochs=20,
             lr=0.01,
             weight_decay=5e-4,
             momentum=0.9,
             patience=5,
             path=None
):
    """
    Train a logistic regression model.
    Args:
        input_dim (int): Input feature dimension.
        num_classes (int): Number of output classes.
        optimizer_type (str): "sgd" or "adam".
        batch_size (int): Batch size.
        epochs (int): Max training epochs.
        lr (float): Learning rate.
        weight_decay (float): Weight decay (L2 regularization).
        momentum (float): Momentum for SGD.
        patience (int): Early stopping patience. 0 disables early stopping.
    """
    # Checkpoint path
    if path is not None:
        ckpt_path = Path(path)
    else:
        ckpt_path = Path("checkpoints/logistic_model_2.pth")

    # Load data
    my_dataloaders, datasets_dict = get_dataloaders("data", batch_size=batch_size, augment=True)

    # Robustly infer input_dim from dataset/dataloader
    if input_dim is None:
        try:
            first_batch = next(iter(my_dataloaders['train']))
            X_batch = first_batch[0]
            # If X_batch is a Tensor, compute per-sample size as product of sample dimensions
            if isinstance(X_batch, torch.Tensor) and X_batch.ndim >= 2:
                per_sample_shape = X_batch.shape[1:]
                input_dim = int(np.prod(per_sample_shape))
        except (StopIteration, KeyError, AttributeError, IndexError, TypeError):
            input_dim = None

    if input_dim is None:
        try:
            sample = datasets_dict['train'][0][0]
            if hasattr(sample, 'numel'):
                input_dim = int(sample.numel())
            else:
                input_dim = int(np.array(sample).ravel().shape[0])
        except Exception as exc:
            raise RuntimeError(
                "Unable to infer input dimension from dataloader/dataset.") from exc

    if num_classes is None:
        # Robustly infer number of classes and class names
        train_dataset = datasets_dict['train']
        if hasattr(train_dataset, 'classes'):
            my_class_names = list(getattr(train_dataset, 'classes'))
            num_classes = len(my_class_names)
        elif hasattr(train_dataset, 'targets'):
            unique = np.unique(np.array(train_dataset.targets))
            num_classes = len(unique)
        else:
            # fallback by scanning a few batches
            label_samples = []
            for _, y in my_dataloaders['train']:
                label_samples.append(y)
                if len(label_samples) >= 5:
                    break
            if label_samples:
                labels_concat = torch.cat(label_samples).numpy()
                unique = np.unique(labels_concat)
                num_classes = len(unique)
            else:
                raise RuntimeError("Unable to infer class labels.")

    my_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize model
    print(f"Input dim: {input_dim}, Num classes: {num_classes}")
    my_model = LogisticRegressionModel(
        input_dim=input_dim, num_classes=num_classes).to(my_device)

    # Optimizer choice
    if optimizer_type == "sgd":
        my_optimizer = torch.optim.SGD(my_model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        my_optimizer = torch.optim.Adam(my_model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0
    best_epoch = 0
    patience = patience if patience > 0 else epochs
    patience_counter = 0

    # Train loop
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            my_model, my_dataloaders['train'], my_optimizer, my_device)
        val_loss, val_acc, _, _, _ = evaluate(
            my_model, my_dataloaders['val'], my_device)
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print("-" * 80)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            save_model(my_model, str(ckpt_path))
            print(
                f"Saved new best checkpoint (epoch {epoch}, val_acc={val_acc:.4f})")
            print("-" * 80)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}. Best val acc: {best_val_acc:.4f} (epoch {best_epoch})")
                break

    print(f"Training finished. Best val acc: {best_val_acc:.4f} (epoch {best_epoch})")
    print("=" * 80)

    return ckpt_path, my_model, my_device, my_dataloaders


if __name__ == "__main__":
    # Configuration constants
    INPUT_DIM = 4096  # Set to None to infer from data
    NUM_CLASSES = 15  # Set to None to infer from data
    OPTIMIZER_TYPE = "sgd"  # "sgd" or "adam"
    BATCH_SIZE = 128
    EPOCHS = 30
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 5e-4
    MOMENTUM = 0.9  # Only used if OPTIMIZER_TYPE is "sgd"
    PATIENCE = 5  # Early stopping patience, set to 0 to disable early stopping

    model_path, the_model, device, the_dataloaders = train_lr(
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
        optimizer_type=OPTIMIZER_TYPE,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        momentum=MOMENTUM,
        patience=PATIENCE
    )

    test(model_path, the_model, device, the_dataloaders)

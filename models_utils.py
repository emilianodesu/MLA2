"""Training and evaluation utilities used across experiments.

Provided utilities
------------------
* ``save_model`` – persist a model ``state_dict`` (parents auto-created)
* ``train_model`` – train + validate with early stopping on validation accuracy
* ``test_model`` – evaluate accuracy, macro‑F1, and confusion matrix
* ``plot_confusion_matrix`` – seaborn heatmap visualization
* ``plot_training_process`` – side‑by‑side loss & accuracy curves
* ``predict_image`` – single image inference (returns predicted label and raw logits)

Conventions
-----------
* Models are expected to output **raw logits** (no softmax). Use ``CrossEntropyLoss`` during training.
* Early stopping monitors validation accuracy (higher is better).
* Returned training metric tuple order from ``train_model`` is: ``(train_acc, train_loss, val_acc, val_loss)``.
* All plotting helpers lazily create parent directories when a save path is provided.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Sequence
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import f1_score, confusion_matrix
from PIL import Image
from preprocessing_utils import get_transforms


def save_model(model: torch.nn.Module, path: str) -> None:
    """
    Save a model's state dict to disk, creating parent folders if needed.

    Args:
        model: The PyTorch module to save.
        path: Destination file path for the ``.pth`` checkpoint.

    Returns:
        None
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def train_model(
    model: torch.nn.Module,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    epochs: int = 20,
    patience: int = 5,
    save_path: str | None = None,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Train a classification model with cross-entropy loss and track metrics over epochs.

    Behavior:
    - Consumes provided dataloaders for the "train" and "val" splits (no dataloader creation inside).
    - Trains for up to ``epochs`` with early stopping on validation accuracy (``patience``).
    - Saves the best checkpoint (by val accuracy) to ``save_path`` or ``checkpoints/model.pth``.
    - Collects training/validation loss and accuracy for plotting.

    Args:
        model: Instantiated model to train.
        dataloaders: Dict with keys "train" and "val" mapping to DataLoader instances.
        optimizer: Torch optimizer (e.g., SGD, Adam) configured with model parameters.
        criterion: Loss function (e.g., ``torch.nn.CrossEntropyLoss()``).
        epochs: Maximum number of epochs to train.
        patience: Early stopping patience measured on validation accuracy (0 disables).
        save_path: Optional path to save the best checkpoint. Defaults to ``checkpoints/model.pth``.

    Returns:
        Tuple[List[float], List[float], List[float], List[float]]:
            (train_acc_per_epoch, train_loss_per_epoch, val_acc_per_epoch, val_loss_per_epoch)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Checkpoint path
    if save_path is not None:
        ckpt_path = Path(save_path)
    else:
        ckpt_path = Path("checkpoints/model.pth")

    # For plotting
    total_loss_train_plot = []
    total_loss_validation_plot = []
    total_acc_train_plot = []
    total_acc_validation_plot = []

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    patience = patience if patience > 0 else epochs

    for epoch in range(1, epochs + 1):
        # ------------------
        # Training phase
        # ------------------
        model.train()
        running_loss_train = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in dataloaders["train"]:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss_train += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_loss_train = (running_loss_train /
                            total_train) if total_train else 0.0
        epoch_acc_train = (correct_train / total_train) if total_train else 0.0

        # ------------------
        # Validation phase
        # ------------------
        model.eval()
        running_loss_val = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs_val, labels_val in dataloaders["val"]:
                inputs_val, labels_val = inputs_val.to(
                    device), labels_val.to(device)

                outputs_val = model(inputs_val)
                val_loss = criterion(outputs_val, labels_val)

                running_loss_val += val_loss.item() * labels_val.size(0)
                preds_val = outputs_val.argmax(dim=1)
                correct_val += (preds_val == labels_val).sum().item()
                total_val += labels_val.size(0)

        epoch_loss_val = (running_loss_val / total_val) if total_val else 0.0
        epoch_acc_val = (correct_val / total_val) if total_val else 0.0

        # ------------------
        # Store for plots
        # ------------------
        total_loss_train_plot.append(round(epoch_loss_train, 4))
        total_loss_validation_plot.append(round(epoch_loss_val, 4))
        total_acc_train_plot.append(round(epoch_acc_train, 4))
        total_acc_validation_plot.append(round(epoch_acc_val, 4))

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {epoch_loss_train:.4f} | Train Acc: {epoch_acc_train:.4f} | "
            f"Val Loss: {epoch_loss_val:.4f} | Val Acc: {epoch_acc_val:.4f}"
        )
        print("-" * 90)

        # ------------------
        # Checkpointing / Early stopping
        # ------------------
        if epoch_acc_val > best_val_acc:
            best_val_acc = epoch_acc_val
            best_epoch = epoch
            patience_counter = 0
            save_model(model, str(ckpt_path))
            print(
                f"Saved new best checkpoint (epoch {epoch}, val_acc={epoch_acc_val:.4f})")
            print("-" * 90)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping at epoch {epoch}. Best val acc: {best_val_acc:.4f} (epoch {best_epoch})")
                break

    print(
        f"Training finished. Best val acc: {best_val_acc:.4f} (epoch {best_epoch})")
    print("=" * 90)

    return (
        total_acc_train_plot,
        total_loss_train_plot,
        total_acc_validation_plot,
        total_loss_validation_plot
    )


@torch.no_grad()
def test_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module | None = None,
) -> Tuple[float, float, np.ndarray]:
    """Evaluate a trained classification model on a dataloader and report metrics.

    Args:
        model: Trained PyTorch model set to evaluation mode during the call.
        dataloader: DataLoader yielding test samples ``(inputs, labels)``.
        criterion: Optional loss function to also report average loss.

    Returns:
        (accuracy, macro_f1, confusion_matrix)

    Notes:
        - If ``criterion`` is provided, average test loss is also printed.
        - Model outputs are assumed to be **logits**; softmax is not applied here.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Track accuracy
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Track loss if criterion provided
        if criterion is not None:
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)

    acc = float(correct / total)
    f1 = float(f1_score(all_labels, all_preds, average="macro"))
    cm = confusion_matrix(all_labels, all_preds)

    if criterion is not None:
        avg_loss = running_loss / total
        print(
            f"Test Loss: {avg_loss:.4f} | Test Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    else:
        print(f"Test Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")

    return acc, f1, cm


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = "Blues",
    save_path: str | None = None,
) -> None:
    """
    Plot a labeled confusion matrix using seaborn heatmap.

    Args:
        cm: Confusion matrix as a 2-D array-like of counts.
        class_names: Class names in the same order as indices used in ``cm``.
        figsize: Figure size in inches.
        cmap: Matplotlib colormap name.
        save_path: If provided, saves the figure to this path (directories are created if needed).

    Returns:
        None. Displays the plot.
    """
    matplotlib.rcParams['font.family'] = "Microsoft YaHei"
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    if save_path is not None:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")


def plot_training_process(
    total_acc_train_plot: Sequence[float],
    total_loss_train_plot: Sequence[float],
    total_acc_validation_plot: Sequence[float],
    total_loss_validation_plot: Sequence[float],
    save_path: str | None = None,
    plot: bool = False,
) -> None:
    """
    Plot loss and accuracy curves for training and validation.

    Args:
        total_acc_train_plot: Training accuracy per epoch.
        total_loss_train_plot: Training loss per epoch.
        total_acc_validation_plot: Validation accuracy per epoch.
        total_loss_validation_plot: Validation loss per epoch.
        save_path: If provided, saves the figure to this path (directories are created if needed).
        plot: If True, also displays the plot inline.

    Returns:
        None. Displays the plot.
    """
    _, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    axs[0].plot(total_loss_train_plot, label='Training Loss')
    axs[0].plot(total_loss_validation_plot, label='Validation Loss')
    axs[0].set_title('Training and Validation Loss over Epochs')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(total_acc_train_plot, label='Training Accuracy')
    axs[1].plot(total_acc_validation_plot, label='Validation Accuracy')
    axs[1].set_title('Training and Validation Accuracy over Epochs')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    plt.tight_layout()

    if save_path is not None:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        plt.savefig(save_path)
        print(f"Training process plot saved to {save_path}")
    if plot:
        plt.show()


def predict_image(
    model: torch.nn.Module,
    image_path: str | Path,
    class_names: List[str],
    image_size: Tuple[int, int] = (64, 64),
    flatten: bool = True,
    normalize: bool = False,
    mean: float = 0.0,
    std: float = 1.0,
    device: torch.device | str = "cpu",
) -> Tuple[str, torch.Tensor]:
    """Preprocess an image and predict its class using a trained model.

    Args:
        model: Trained PyTorch model (already loaded with weights).
        image_path: Path to the input image (jpg/png).
        class_names: List of class labels corresponding to output indices.
        image_size: Size to resize image (default 64x64).
        flatten: If True, flatten image into vector (for linear models).
        normalize: Apply normalization if True.
        mean, std: Stats for normalization.
        device: "cpu" or "cuda".

    Returns:
        tuple: (predicted_class_name, logits_tensor)

    Notes:
        The returned tensor is **raw logits** of shape ``(1, num_classes)``. Apply
        ``torch.softmax(logits, dim=1)`` if probabilities are required.
    """
    # Resolve device and set eval mode
    device = torch.device(device) if isinstance(device, str) else device
    model.eval()
    model.to(device)

    # Same transform as test set
    transform = get_transforms(
        image_size=image_size,
        augment=False,
        flatten=flatten,
        normalize=normalize,
        mean=mean,
        std=std
    )

    # Load and preprocess image
    with Image.open(image_path) as img:
        plt.imshow(img)
        plt.show()
        x_tensor = transform(img)  # transformed image as tensor
        if not isinstance(x_tensor, torch.Tensor):
            raise TypeError("Transform pipeline must return a torch.Tensor. Please ensure transforms.ToTensor() is included.")
        x = x_tensor.unsqueeze(0).to(device)  # add batch dim

    # Prediction
    with torch.no_grad():
        outputs = model(x)
        pred_idx = int(outputs.argmax().item())

    # Optional sanity check: number of classes vs probabilities length
    if len(class_names) != outputs.numel():
        print(
            f"Warning: class_names length ({len(class_names)}) != model outputs ({outputs.numel()})."
        )

    return class_names[pred_idx], outputs

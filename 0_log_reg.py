"""
Logistic Regression Training and Evaluation Script

This script trains and evaluates a logistic regression model on the Chinese MNIST dataset.
It supports robust input dimension and class inference, early stopping, checkpointing, and test evaluation.

Usage:
    - Edit the configuration constants in the __main__ block to change behavior.
    - Set TEST_ONLY = True to skip training and only evaluate a saved checkpoint.
    - Set RUN_TEST = True to evaluate the best model on the test set after training.
"""
import torch
from models import LogisticRegressionModel
from utils import get_dataloaders, save_model
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix
from pathlib import Path
import numpy as np


def _flatten_if_needed(x):
    """
    Flatten a batched tensor to shape (batch, -1) if it's multi-dimensional (C,H,W).
    Leaves (batch, features) untouched.

    Args:
        x (torch.Tensor): Input tensor of shape (batch, ...) or (batch, features).

    Returns:
        torch.Tensor: Flattened tensor if needed, otherwise original tensor.
    """
    if isinstance(x, torch.Tensor) and x.ndim > 2:
        return x.view(x.size(0), -1)
    return x


def train_one_epoch(model, dataloader, optimizer, device):
    """
    Run a single training epoch.

    Args:
        model (torch.nn.Module): Model in train mode.
        dataloader (DataLoader): Yields (inputs, targets).
        optimizer (torch.optim.Optimizer): Optimizer to step.
        device (str): Device string ("cpu" or "cuda").

    Returns:
        tuple: (avg_loss (float), accuracy (float)). Returns (0.0, 0.0) if no samples.
    """
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        X = _flatten_if_needed(X)
        optimizer.zero_grad()
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_loss += loss.item() * y.size(0)
        total_samples += y.size(0)

    if total_samples == 0:
        return 0.0, 0.0

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    Evaluate model on a dataloader and return average loss and accuracy.

    Args:
        model (torch.nn.Module): Model in eval mode.
        dataloader (DataLoader): Yields (inputs, targets).
        device (str): Device string.

    Returns:
        tuple: (avg_loss (float), accuracy (float)). Returns (0.0, 0.0) if no samples.
    """
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        X = _flatten_if_needed(X)
        logits = model(X)
        loss = F.cross_entropy(logits, y)

        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_loss += loss.item() * y.size(0)
        total_samples += y.size(0)
    if total_samples == 0:
        return 0.0, 0.0
    return total_loss / total_samples, total_correct / total_samples


def evaluate_full(model, dataloader, device):
    """
    Evaluate model on a dataloader and return multiple metrics in a single pass.

    Args:
        model (torch.nn.Module): Model in eval mode.
        dataloader (DataLoader): Yields (inputs, targets).
        device (str): Device string.

    Returns:
        tuple: (
            avg_loss (float),
            accuracy (float),
            macro_f1 (float),
            confusion_matrix (np.ndarray),
            labels_order (np.ndarray)
        )
        Returns zeros/empty arrays if no samples.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_targets = []

    with (torch.no_grad()):
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X = _flatten_if_needed(X)
            logits = model(X)
            loss = F.cross_entropy(logits, y)

            preds = logits.argmax(dim=1)

            total_correct += (preds == y).sum().item()
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())

    if total_samples == 0:
        return 0.0, 0.0, 0.0, np.zeros((0, 0), dtype=int), []

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    my_labels_order = np.unique(np.concatenate([all_targets, all_preds]))
    my_cm = confusion_matrix(all_targets, all_preds, labels=my_labels_order)

    return avg_loss, acc, macro_f1, my_cm, my_labels_order


if __name__ == "__main__":
    # Configuration. Edit these constants to change behavior.
    BATCH_SIZE = 64
    EPOCHS = 20
    LR = 0.01
    WEIGHT_DECAY = 5e-4
    OPTIMIZER = "sgd"  # choose "adam" or "sgd"
    RUN_TEST = True  # Whether to run test evaluation after training
    TEST_ONLY = False  # If True, skip training and run test only (requires existing checkpoint)
    PATIENCE = 5

    CKPT_PATH = Path("checkpoints/logistic_model.pth")

    # Load data
    dataloaders, datasets_dict = get_dataloaders("data", batch_size=BATCH_SIZE, augment=True)

    # Robustly infer input_dim from dataset/dataloader
    input_dim = None
    try:
        first_batch = next(iter(dataloaders['train']))
        X_batch = first_batch[0]
        # If X_batch is a Tensor, compute per-sample size as product of sample dimensions
        if isinstance(X_batch, torch.Tensor) and X_batch.ndim >= 2:
            per_sample_shape = X_batch.shape[1:]
            input_dim = int(np.prod(per_sample_shape))
    except Exception:
        input_dim = None

    if input_dim is None:
        try:
            sample = datasets_dict['train'][0][0]
            if hasattr(sample, 'numel'):
                input_dim = int(sample.numel())
            else:
                input_dim = int(np.array(sample).ravel().shape[0])
        except Exception:
            raise RuntimeError(
                "Unable to infer input dimension from dataloader/dataset. Ensure dataset returns tensors.")

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
        for _, y in dataloaders['train']:
            label_samples.append(y)
            if len(label_samples) >= 5:
                break
        if label_samples:
            labels_concat = torch.cat(label_samples).numpy()
            unique = np.unique(labels_concat)
            num_classes = len(unique)
        else:
            raise RuntimeError(
                "Unable to infer class labels. Provide a dataset with `classes` or `targets` or non-empty dataloader.")

    my_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize model
    my_model = LogisticRegressionModel(input_dim=input_dim, num_classes=num_classes).to(my_device)

    # If TEST_ONLY, load checkpoint and run test evaluation only
    if TEST_ONLY:
        if not CKPT_PATH.exists():
            print(f"Checkpoint {CKPT_PATH} not found. Cannot run test-only mode.")
        else:
            my_model.load_state_dict(torch.load(CKPT_PATH, map_location=my_device))
            print(f"Loaded checkpoint {CKPT_PATH} for test-only evaluation.")
            test_loss, test_acc, test_f1, cm, labels_order = evaluate_full(my_model, dataloaders['test'], my_device)
            print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f} | Test Macro F1: {test_f1:.4f}")
            print(cm)
        raise SystemExit(0)

    # Optimizer choice
    if OPTIMIZER == "sgd":
        my_optimizer = torch.optim.SGD(my_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, momentum=0.9)
    else:
        my_optimizer = torch.optim.Adam(my_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_acc = 0.0
    best_epoch = 0
    patience = PATIENCE
    patience_counter = 0

    # Train loop
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(my_model, dataloaders['train'], my_optimizer, my_device)
        val_loss, val_acc = evaluate(my_model, dataloaders['val'], my_device)
        print(
            f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            save_model(my_model, str(CKPT_PATH))
            print(f"Saved new best checkpoint (epoch {epoch}, val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}. Best val acc: {best_val_acc:.4f} (epoch {best_epoch})")
                break

    print(f"Training finished. Best val acc: {best_val_acc:.4f} (epoch {best_epoch})")

    # After training, if RUN_TEST is True, load best model and evaluate on test set
    if RUN_TEST:
        if not CKPT_PATH.exists():
            print(f"Checkpoint {CKPT_PATH} not found, skipping test evaluation.")
        else:
            my_model.load_state_dict(torch.load(CKPT_PATH, map_location=my_device))
            print(f"Loaded checkpoint {CKPT_PATH} for test evaluation.")

            test_loss, test_acc, test_f1, cm, labels_order = evaluate_full(my_model, dataloaders['test'], my_device)

            print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f} | Test Macro F1: {test_f1:.4f}")
            print(cm)
    else:
        print("Test evaluation skipped. Set RUN_TEST = True to enable it.")

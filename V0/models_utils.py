"""
Various utilities for training/evaluation.
"""
import os
import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix

# from preprocessing_utils import flatten_if_needed


def save_model(model, path: str):
    """
    Save model state dict to a file.
    Args:
        model: torch.nn.Module
        path: str, file path where to save the model
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


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
        # X = flatten_if_needed(X)
        optimizer.zero_grad()
        logits = model(X)
        loss = torch.nn.functional.cross_entropy(logits, y)
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
    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_preds = []
    all_targets = []

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        # X = flatten_if_needed(X)
        logits = model(X)
        
        loss = torch.nn.functional.cross_entropy(logits, y)
        total_loss += loss.item() * y.size(0)

        preds = logits.argmax(dim=1)

        total_correct += (preds == y).sum().item()
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
    labels_order = np.unique(np.concatenate([all_targets, all_preds]))
    cm = confusion_matrix(all_targets, all_preds, labels=labels_order)

    return avg_loss, acc, macro_f1, cm, labels_order


def test(ckpt_path, model, device, dataloaders):
    """
    Load model from checkpoint and evaluate on test set.
    Args:
        ckpt_path (Path): Path to model checkpoint.
        model (torch.nn.Module): Model instance.
        device (str): Device string.
        dataloaders (dict): Dictionary with 'test' DataLoader.
    """
    if not ckpt_path.exists():
        print(f"Checkpoint {ckpt_path} not found, skipping test evaluation.")
    else:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded checkpoint {ckpt_path} for test evaluation.")

        test_loss, test_acc, test_f1, cm, _ = evaluate(
            model, dataloaders['test'], device)

        print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f} | Test Macro F1: {test_f1:.4f}")
        print(cm)

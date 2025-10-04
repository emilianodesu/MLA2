"""Logistic Regression training script for Chinese MNIST.

Overview
--------
This script trains a **single linear (logistic regression) classifier** on the flattened
64×64 grayscale Chinese MNIST images. It explores a small hyperparameter grid across
batch size, optimizer, and learning rate, evaluates all saved checkpoints on the test set,
then reports and visualizes the best performer.

Pipeline Steps
--------------
1. Dataset preparation / validation via :func:`preprocessing_utils.prepare_images`.
2. Construction of train / validation / test DataLoaders (optionally with light augmentation
     on the training split) via :func:`preprocessing_utils.get_dataloaders`.
3. Definition of a single-layer linear classifier (:class:`LogisticRegressionModel`).
4. Training across the grid:
     - batch_size: ``[32, 256]`` (NOTE: The earlier 64 value was removed to reduce runtime / memory.)
     - optimizer: ``Adam`` and ``SGD`` (SGD uses momentum=0.9)
     - learning_rate: ``[0.01, 0.001, 0.0005]`` (updated; ``0.005`` no longer used)
     - epochs: 30 (with early stopping)
     - early stopping patience: 5 epochs (monitors validation accuracy)
5. Checkpoint saving: best validation accuracy per run →
     ``checkpoints/log_reg/model_{batch}_{optimizer}_{lr}.pth``
6. Curve plotting: training / validation accuracy & loss → ``plots/log_reg/``
7. Test evaluation of every checkpoint (accuracy, macro‑F1, confusion matrix) to find best model.
8. Confusion matrix plotting for the best checkpoint.
9. Simple per‑class qualitative inference (one random test image per class) using the best model.

Key Assumptions
---------------
- Inputs passed to :class:`LogisticRegressionModel` are already **flattened** to shape
    ``(batch_size, input_dim)``. The module does **not** perform flattening internally.
- ``input_dim`` is inferred dynamically from a sample batch after the initial DataLoader creation.
- Utilities sourced from ``preprocessing_utils.py`` and ``models_utils.py`` (training loop,
    evaluation, plotting, prediction helper).
- Returned values from :func:`predict_image` are raw logits (not softmax probabilities). When
    the script prints them it labels them as *probabilities* for convenience; apply
    ``torch.softmax(logits, dim=0)`` if true probabilities are required.

Run
---
        python log_reg.py

Reproducibility / Notes
-----------------------
- Random sampling (image previews, per-class inference) is not seeded; set a manual seed if
    deterministic behavior is needed.
- Early stopping triggers when validation accuracy fails to improve for ``patience`` epochs;
    best checkpoint is preserved.
"""
import os
import re
import random
import torch
from torchinfo import summary
from preprocessing_utils import prepare_images, get_dataloaders, summarize_split, preview_random_images, show_batch
from models_utils import train_model, test_model, plot_confusion_matrix, plot_training_process, predict_image


class LogisticRegressionModel(torch.nn.Module):
    """
    Baseline 0: Logistic Regression for Chinese Characters.

    A single linear layer mapping `input_dim -> num_classes` trained with cross-entropy.
    Note: This module does not apply any activation or flattening internally.

    Args:
        input_dim (int): Number of input features per sample (e.g., 64*64 for a 1x64x64 image if flattened).
        num_classes (int): Number of target classes.
    """

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, num_classes)

    def forward(self, x):
        """
        Compute logits for a batch.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch_size, input_dim)`. Must be 2D and already flattened.

        Returns:
            torch.Tensor: Logits of shape `(batch_size, num_classes)`.
        """
        return self.linear(x)


if __name__ == "__main__":

    # Check PyTorch version
    print(torch.__version__)
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Getting images ready
    # Prepare dataset
    root = prepare_images()
    print(f"Dataset ready at: {root}")
    # Summarize dataset
    summarize_split(str(root))
    # Preview random images
    preview_random_images(data_dir=str(root / "c_mnist"), n_images=9, grid_size=(3, 3))

    # Preprocessing
    # Get dataloaders
    dataloaders, datasets = get_dataloaders(data_dir="data", batch_size=64, image_size=(64, 64), augment=True)
    labels = datasets['train'].classes
    print("DataLoaders ready.")
    print(f"Classes: {labels}")

    show_batch(datasets["train"], n=12, cols=4)

    # Hyperparameters
    sample_batch = next(iter(dataloaders["train"]))[0]
    INPUT_DIM = sample_batch.shape[1] if sample_batch.ndim == 2 else sample_batch[0].numel()
    NUM_CLASSES = len(labels)
    # Batch sizes explored (kept small & large for gradient noise vs. stability comparison)
    BATCH_SIZE = [32, 256]
    EPOCHS = 30
    LEARNING_RATE = [0.01, 0.001, 0.0005]
    WEIGHT_DECAY = 5e-4
    MOMENTUM = 0.9  # For SGD
    PATIENCE = 5

    # Model definition
    log_reg_model = LogisticRegressionModel(input_dim=INPUT_DIM, num_classes=NUM_CLASSES).to(device)

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    # Model summary
    print(f"Logistic Regression summary (input_dim={INPUT_DIM}, num_classes={NUM_CLASSES})")
    summary(log_reg_model, input_size=(1, INPUT_DIM), col_names=("input_size", "output_size", "num_params"))

    # Training
    dataloaders_dict = {}
    for bs in BATCH_SIZE:
        dl, _ = get_dataloaders(
            data_dir="data", batch_size=bs, image_size=(64, 64), augment=True)
        dataloaders_dict[bs] = dl
        print(f"DataLoaders for batch size {bs} ready.")

    total_acc_train_sgd = {}
    total_acc_validation_sgd = {}
    total_loss_train_sgd = {}
    total_loss_validation_sgd = {}

    total_acc_train_adam = {}
    total_acc_validation_adam = {}
    total_loss_train_adam = {}
    total_loss_validation_adam = {}

    for bs in BATCH_SIZE:
        print(f"\nTraining with batch size: {bs}")
        dataloaders = dataloaders_dict[bs]
        run_idx = 1
        for lr in LEARNING_RATE:
            # Adam run
            print(f"  Run {run_idx}: Adam optimizer with lr={lr}")
            model_adam = LogisticRegressionModel(INPUT_DIM, NUM_CLASSES).to(device)
            adam_opt = torch.optim.Adam(model_adam.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
            (
                acc_train_adam,
                loss_train_adam,
                acc_val_adam,
                loss_val_adam,
            ) = train_model(
                model_adam,
                dataloaders,
                adam_opt,
                criterion,
                epochs=EPOCHS,
                patience=PATIENCE,
                save_path=f"checkpoints/log_reg/model_{bs}_adam_{lr}.pth",
            )
            total_acc_train_adam[(bs, lr)] = acc_train_adam
            total_loss_train_adam[(bs, lr)] = loss_train_adam
            total_acc_validation_adam[(bs, lr)] = acc_val_adam
            total_loss_validation_adam[(bs, lr)] = loss_val_adam

            # SGD run
            print(f"  Run {run_idx}: SGD optimizer with lr={lr}, momentum={MOMENTUM}")
            model_sgd = LogisticRegressionModel(INPUT_DIM, NUM_CLASSES).to(device)
            sgd_opt = torch.optim.SGD(model_sgd.parameters(), lr=lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
            (
                acc_train_sgd,
                loss_train_sgd,
                acc_val_sgd,
                loss_val_sgd,
            ) = train_model(
                model_sgd,
                dataloaders,
                sgd_opt,
                criterion,
                epochs=EPOCHS,
                patience=PATIENCE,
                save_path=f"checkpoints/log_reg/model_{bs}_sgd_{lr}.pth",
            )
            total_acc_train_sgd[(bs, lr)] = acc_train_sgd
            total_loss_train_sgd[(bs, lr)] = loss_train_sgd
            total_acc_validation_sgd[(bs, lr)] = acc_val_sgd
            total_loss_validation_sgd[(bs, lr)] = loss_val_sgd

            run_idx += 1

    # Plotting training process
    for bs in BATCH_SIZE:
        for lr in LEARNING_RATE:
            print(f"\nPlotting training process for batch size {bs} and learning rate {lr} (SGD)")
            plot_training_process(
                total_acc_train_sgd[(bs, lr)],
                total_loss_train_sgd[(bs, lr)],
                total_acc_validation_sgd[(bs, lr)],
                total_loss_validation_sgd[(bs, lr)],
                f"plots/log_reg/training_process_{bs}_{lr}_sgd.png",
                plot=False
            )

            print(f"\nPlotting training process for batch size {bs} and learning rate {lr} (Adam)")
            plot_training_process(
                total_acc_train_adam[(bs, lr)],
                total_loss_train_adam[(bs, lr)],
                total_acc_validation_adam[(bs, lr)],
                total_loss_validation_adam[(bs, lr)],
                f"plots/log_reg/training_process_{bs}_{lr}_adam.png",
                plot=False
            )

    # Testing
    MODELS_DIR = "checkpoints/log_reg"

    best = {
        "path": None,
        "acc": 0.0,
        "f1": 0.0,
        "cm": None
    }

    # Iterate over all .pth files in directory
    if os.path.isdir(MODELS_DIR):
        for model_file in os.listdir(MODELS_DIR):
            if not model_file.endswith(".pth"):
                continue

            model_path = os.path.join(MODELS_DIR, model_file)
            print(f"\nTesting model: {model_path}")

            # Extract batch size from filename (model_32_adam_0.01.pth → 32)
            match = re.search(r"model_(\d+)_", model_file)
            if not match:
                print(f"Could not extract batch size from {model_file}, skipping...")
                continue
            batch_size = int(match.group(1))

            # Load model
            log_reg_model = LogisticRegressionModel(INPUT_DIM, NUM_CLASSES).to(device)
            log_reg_model.load_state_dict(torch.load(model_path, map_location=device))
            log_reg_model.eval()

            # Evaluate with correct dataloader
            test_loader = dataloaders_dict.get(batch_size, dataloaders_dict[BATCH_SIZE[0]])["test"]
            test_acc, f1, cm = test_model(log_reg_model, test_loader, criterion)

            print(f"Accuracy: {test_acc:.4f} | F1: {f1:.4f}")

            # Update best model
            if test_acc > best["acc"]:
                best.update({"path": model_path, "acc": test_acc, "f1": f1, "cm": cm})
                print(f"New best model: {model_path} (Acc: {test_acc:.4f}, F1: {f1:.4f})")
    else:
        print(f"Models directory not found: {MODELS_DIR}")

    # Plot training process and confusion matrix of the best model
    if best["cm"] is not None:
        print(f"\nBest model: {best['path']} | Acc: {best['acc']:.4f} | F1: {best['f1']:.4f}")
        plot_confusion_matrix(best["cm"], datasets["train"].classes)

    # Extract batch size and learning rate from best model filename
    m = re.search(r"model_(\d+)_(adam|sgd)_([0-9.]+)\.pth", os.path.basename(best["path"]))
    if m:
        batch_size = int(m.group(1))
        optimizer = m.group(2)
        learning_rate = float(m.group(3))
        print(f"Extracted batch size: {batch_size}, optimizer: {optimizer}, learning rate: {learning_rate}")
        if optimizer == "sgd":
            plot_training_process(
                total_acc_train_sgd[(batch_size, learning_rate)],
                total_loss_train_sgd[(batch_size, learning_rate)],
                total_acc_validation_sgd[(batch_size, learning_rate)],
                total_loss_validation_sgd[(batch_size, learning_rate)],
                f"plots/log_reg/best_model_training_process_{batch_size}_{learning_rate}_{optimizer}.png"
            )
        else:
            plot_training_process(
                total_acc_train_adam[(batch_size, learning_rate)],
                total_loss_train_adam[(batch_size, learning_rate)],
                total_acc_validation_adam[(batch_size, learning_rate)],
                total_loss_validation_adam[(batch_size, learning_rate)],
                f"plots/log_reg/best_model_training_process_{batch_size}_{learning_rate}_{optimizer}.png"
            )
    else:
        print("Could not extract parameters from best model filename. Skipping best-run plot.")

    # Inference on random images from test set
    # Collect one random image per class folder in data/test
    TEST_ROOT = "data/test"
    selected_images = []
    for cls in sorted(os.listdir(TEST_ROOT)):
        cls_dir = os.path.join(TEST_ROOT, cls)
        if not os.path.isdir(cls_dir):
            continue
        imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg",))]
        if not imgs:
            continue
        selected = os.path.join(cls_dir, random.choice(imgs))
        selected_images.append(selected)

    if best["path"]:
        log_reg_model = LogisticRegressionModel(INPUT_DIM, NUM_CLASSES).to(device)
        log_reg_model.load_state_dict(torch.load(best["path"], map_location=device))
        log_reg_model.eval()

        for img_path in selected_images:
            print(f"Selected image for prediction: {img_path}")
            prediction, logits = predict_image(
                model=log_reg_model,
                image_path=img_path,
                class_names=datasets['train'].classes,
                device=device,
            )
            # NOTE: 'logits' are raw scores; apply torch.softmax(logits, dim=0) for probabilities
            print(f"Predicted class: {prediction} | Logits: {logits}")
    else:
        print("No best model available; skipping inference.")

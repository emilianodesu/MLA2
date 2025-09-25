import os
import shutil
import random
import requests
import zipfile
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def prepare_images(
        url="https://github.com/emilianodesu/MLA2/raw/main/data.zip",
        data_dir="data",
        split_ratio=(0.8, 0.1, 0.1),
        overwrite=False,
        label_type="value"
):
    """
    Download, extract, and prepare the Chinese MNIST dataset.
    Splits the dataset into train/val/test folders according to the given ratio.

    Args:
        url (str): URL to download the dataset zip file.
        data_dir (str): Directory to store the dataset.
        split_ratio (tuple): Tuple of (train, val, test) split ratios. Must sum to 1.
        overwrite (bool): If True, overwrite existing split folders.
        label_type (str): Label type for classification ('value', 'value_character', or 'code').
    Returns:
        Path: Path object to the data directory.
    """
    data_path = Path(data_dir)
    image_path = data_path / "c_mnist"
    csv_path = data_path / "chinese_mnist.csv"
    zip_path = "data.zip"

    # Check existence of dataset
    if image_path.is_dir() and csv_path.is_file():
        print("Dataset already exists, skipping download.")
    else:
        print("Dataset not found, downloading...")
        data_path.mkdir(parents=True, exist_ok=True)

        # Download
        response = requests.get(url, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")

        # Extract
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall("./")
        print("Extraction complete.")

    # Check if split already exists
    train_dir, val_dir, test_dir = data_path / "train", data_path / "val", data_path / "test"
    if (train_dir.exists() or val_dir.exists() or test_dir.exists()) and not overwrite:
        print("Train/Val/Test folders already exist. Skipping split. (Set overwrite=True to recreate them)")
        return data_path

    # If overwrite is enabled, clean old folders
    if overwrite:
        for folder in [train_dir, val_dir, test_dir]:
            if folder.exists():
                shutil.rmtree(folder)
                print(f"Removed existing {folder}")

    # Split dataset into train/val/test
    df = pd.read_csv(csv_path)
    df["filename"] = df.apply(
        lambda row: f"input_{row.suite_id}_{row.sample_id}_{row.code}.jpg", axis=1
    )
    # Choose label type: "code" (default), "value", or "value_character"
    if label_type == "value":
        df["class"] = df["value"].astype(str)
    elif label_type == "value_character":
        df["class"] = df["value"].astype(str) + "_" + df["character"]
    else:  # "code"
        df["class"] = df["code"].astype(str)

    # Validate ratios
    train_ratio, val_ratio, test_ratio = split_ratio
    assert abs(sum(split_ratio) - 1.0) < 1e-6, "Split ratios must sum to 1."

    # Train/Test/Val split
    df_train, df_temp = train_test_split(df, test_size=(1 - train_ratio), stratify=df["class"], random_state=42)
    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    df_val, df_test = train_test_split(df_temp, test_size=(1 - relative_val_ratio), stratify=df_temp["class"],
                                       random_state=42)

    # Helper to copy files
    def copy_files(subset_df, split_name):
        split_dir = data_path / split_name
        for _, row in subset_df.iterrows():
            src = image_path / row["filename"]
            dst_dir = split_dir / row["class"]
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst_dir / row["filename"])

    print("Creating train/val/test folders...")
    copy_files(df_train, "train")
    copy_files(df_val, "val")
    copy_files(df_test, "test")

    print("Data preparation complete.")
    return data_path


def summarize_split(base_dir="data"):
    """
    Print a summary of the number of images and classes in each split (train/val/test).

    Args:
        base_dir (str): Base directory containing the split folders.
    """
    base = Path(base_dir)
    for split in ["train", "val", "test"]:
        split_dir = base / split
        if not split_dir.exists():
            print(f"{split_dir} missing")
            continue
        classes = [p.name for p in split_dir.iterdir() if p.is_dir()]
        counts = {c: len(list((split_dir / c).glob("*.jpg"))) for c in classes}
        total = sum(counts.values())
        print(f"{split}: {total} images, {len(classes)} classes")
        print("  sample:", dict(list(counts.items())[:5]))


def preview_random_images(data_dir="data/c_mnist", n_images=9, grid_size=(3, 3)):
    """
    Preview random images from the Chinese MNIST dataset.

    Args:
        data_dir (str or Path): Path to the image folder (before split).
        n_images (int): Total number of random images to show.
        grid_size (tuple): Grid layout (rows, cols).
    Raises:
        FileNotFoundError: If no .jpg images are found in the directory.
    """
    data_path = Path(data_dir)
    all_images = list(data_path.glob("*.jpg"))

    if len(all_images) == 0:
        raise FileNotFoundError(f"No .jpg images found in {data_path}")

    # Pick random images
    sample_images = random.sample(all_images, min(n_images, len(all_images)))

    # Setup plot
    fig, axes = plt.subplots(*grid_size, figsize=(10, 10))
    axes = axes.flatten()

    for img_path, ax in zip(sample_images, axes):
        img = Image.open(img_path)
        img_array = np.array(img)
        h, w, c = img_array.shape if img_array.ndim == 3 else (*img_array.shape, 1)

        # Class info from filename (input_suite_sample_code.jpg → use code)
        parts = img_path.stem.split("_")
        code_id = int(parts[-1])

        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Class ID: {code_id}\nShape: [{h}, {w}, {c}]",
                     fontsize=9)

    # Hide unused subplots if n_images < rows*cols
    for ax in axes[len(sample_images):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def get_transforms(image_size=(64, 64), augment=True, flatten=True):
    """
    Returns torchvision transforms for train/val/test.

    Args:
        image_size (tuple): Resize images to (H,W)
        augment (bool): Apply light augmentation (train only)
        flatten (bool): Flatten image to 1D tensor

    Returns:
        torchvision.transforms.Compose: Composed transform for image preprocessing.
    """

    transform_list = [transforms.Grayscale(num_output_channels=1),
                      transforms.Resize(image_size)]

    if augment:
        transform_list.extend([transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.05, 0.05))
        ])

    transform_list.append(transforms.ToTensor())  # [0,1]

    if flatten:
        transform_list.append(transforms.Lambda(lambda x: x.view(-1)))  # flatten

    return transforms.Compose(transform_list)


def get_dataloaders(data_dir="data",
                    batch_size=64,
                    num_workers=0,
                    pin_memory=False,
                    image_size=(64, 64),
                    augment=True,
                    flatten=True):
    """
    Returns train/val/test DataLoaders for Chinese MNIST.
    Args:
        data_dir (str): Folder containing train/val/test splits.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of DataLoader workers.
        pin_memory (bool): DataLoader pin_memory option.
        image_size (tuple): Resize images to this size.
        augment (bool): Apply train augmentation.
        flatten (bool): Flatten images to 1D tensors.
    Returns:
        tuple: (dict of DataLoaders, dict of Datasets)
    """
    # Transforms
    train_transform = get_transforms(image_size=image_size, augment=augment, flatten=flatten)
    test_transform = get_transforms(image_size=image_size, augment=False, flatten=flatten)

    datasets_dict = {}
    dataloaders_dict = {}

    for split in ["train", "val", "test"]:
        split_path = f"{data_dir}/{split}"
        transform = train_transform if split == "train" else test_transform
        dataset = datasets.ImageFolder(root=split_path, transform=transform)
        datasets_dict[split] = dataset

        shuffle = True if split == "train" else False
        dataloaders_dict[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory
        )

    return dataloaders_dict, datasets_dict


def show_batch(dataset, class_names=None, n=16, cols=4, title="Sample images"):
    """
    Visualize a batch of images (or PCA features if used).

    Args:
        dataset: PyTorch Dataset (ImageFolder or TensorDataset)
        class_names: list of class names (optional, use index if None)
        n: number of samples to display
        cols: number of grid columns
        title: overall plot title
    """
    idxs = np.random.choice(len(dataset), size=n, replace=False)
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(cols * 3, rows * 3))
    plt.suptitle(title, fontsize=14)

    for i, idx in enumerate(idxs):
        x, y = dataset[idx]
        plt.subplot(rows, cols, i + 1)

        if isinstance(x, torch.Tensor) and x.ndim == 1:
            # Flattened vector (MLP/PCA case)
            size = int(np.sqrt(x.shape[0])) if np.sqrt(x.shape[0]).is_integer() else None
            if size:
                img = x.view(size, size).numpy()
                plt.imshow(img, cmap="gray")
            else:
                plt.plot(x.numpy())  # fallback: plot vector
        else:
            # Regular image tensor (C,H,W)
            if x.ndim == 3:
                img = x.squeeze(0).numpy()  # (1,H,W) → (H,W)
                plt.imshow(img, cmap="gray")
            else:
                plt.imshow(x, cmap="gray")

        # Show numeric label only
        label = f"Class {y}"

        plt.title(f"{label}\nShape: {tuple(x.shape)}", fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


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


def load_model(model_class, input_dim: int, num_classes: int, path: str, device: str = "cpu"):
    """
    Load a model from file and return it.
    Args:
        model_class: the model class (e.g., LogisticRegressionModel)
        input_dim: int, input dimension of the model
        num_classes: int, number of classes
        path: str, file path where the model is saved
        device: "cpu" or "cuda"
    Returns:
        model: loaded model instance
    """
    model = model_class(input_dim=input_dim, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"Model loaded from {path}")
    return model


if __name__ == "__main__":
    root = prepare_images()
    print(f"Dataset ready at: {root}")
    summarize_split(str(root))
    preview_random_images(data_dir=root / "c_mnist", n_images=9, grid_size=(3, 3))

    # Get DataLoaders
    dataloaders, datasets = get_dataloaders(data_dir="data", batch_size=64, image_size=(64, 64), augment=True)
    print("DataLoaders ready.")

    show_batch(datasets['train'], class_names=datasets['train'].classes, n=12, cols=4)

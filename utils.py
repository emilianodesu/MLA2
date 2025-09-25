import shutil
import requests
import zipfile
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def prepare_chinese_mnist(
        url="https://github.com/emilianodesu/MLA2/raw/main/data.zip",
        data_dir="data",
        split_ratio=(0.8, 0.1, 0.1),
        overwrite=False
):
    """
    Download, extract and split the Chinese MNIST dataset into train/val/test folders.

    Args:
        url (str): URL to download data.zip if not already available.
        data_dir (str): Base folder where data will be stored.
        split_ratio (tuple): Train/Val/Test split ratio. Should sum to 1.
        overwrite (bool): If True, will overwrite existing train/val/test folders.

    Returns:
        Path: Path to the dataset base directory.
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
    df["class"] = df["value"].astype(str) + "_" + df["character"]

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


if __name__ == "__main__":
    root = prepare_chinese_mnist()
    print(f"Dataset ready at: {root}")

import gzip
from pathlib import Path

import numpy as np
import pandas as pd


def transform_tabzilla_to_csv(tabzilla_dir: str, output_dir: str) -> None:
    """
    Transforms TabZilla data format into train/test CSV files.

    Args:
        tabzilla_dir (str): Path to the TabZilla dataset directory.
        output_dir (str): Path to the output directory where train/test folders will be created.
    """
    tabzilla_dir = Path(tabzilla_dir)
    output_dir = Path(output_dir)
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for dataset_dir in tabzilla_dir.iterdir():
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name
        metadata_path = dataset_dir / "metadata.json"
        X_path = dataset_dir / "X.npy.gz"
        y_path = dataset_dir / "y.npy.gz"
        split_indices_path = dataset_dir / "split_indeces.npy.gz"

        if not all(
            path.exists()
            for path in [metadata_path, X_path, y_path, split_indices_path]
        ):
            print(f"Skipping {dataset_name}: Missing required files.")
            continue

        with gzip.open(X_path, "rb") as f:
            X = np.load(f, allow_pickle=True)
        with gzip.open(y_path, "rb") as f:
            y = np.load(f, allow_pickle=True)
        with gzip.open(split_indices_path, "rb") as f:
            split_indices = np.load(f, allow_pickle=True)

        data = pd.DataFrame(X)
        data["target"] = y

        train_indices = split_indices[0]["train"]
        test_indices = split_indices[0]["test"]

        train_data = data.iloc[train_indices]
        test_data = data.iloc[test_indices]

        train_data.to_csv(train_dir / f"{dataset_name}.csv", index=False)
        test_data.to_csv(test_dir / f"{dataset_name}.csv", index=False)

        print(f"Processed dataset: {dataset_name}")


if __name__ == "__main__":
    transform_tabzilla_to_csv(
        tabzilla_dir="resources/tabzilla/datasets",
        output_dir="resources/data/openml",
    )

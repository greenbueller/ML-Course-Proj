from pathlib import Path
from src.datasets.kaggle_auth import get_kaggle_api

DATASET_ID = "crawford/cat-dataset"

def download(destination: Path) -> None:
    """
    Download the Cat dataset from Kaggle.

    Args:
        destination (Path): The destination directory to save the dataset.
    """
    destination.mkdir(parents=True, exist_ok=True)

    api = get_kaggle_api()

    print(f"Downloading dataset '{DATASET_ID}' to '{destination}'...")

    api.dataset.download_files(
        DATASET_ID,
        path=str(destination),
        unzip=True,
    )

    print(f"Dataset '{DATASET_ID}' downloaded and extracted to '{destination}'.")
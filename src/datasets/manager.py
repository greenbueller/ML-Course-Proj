from pathlib import Path
from . import catloaf
from . import breeds
from . import cats
import os
import stat

# Cache Config

# Stored in your local user files to avoid accidentally doing a massive commit
# Ex:
#       Windows: C:/Users/[you]/loaf-cache
#       Mac: /Users/[you]/loaf-cache
#       Linux: /home/[you]/loaf-cache
CACHE_DIR = Path.home() / "loaf-cache"

# Datasets

DATASETS = {
    "set_1": {
        "name": "Cat vs Cat-Loaf",
        "module": catloaf,
        "directory": "set_1",
    },
    "set_2": {
        "name": "Cat Breeds",
        "module": breeds,
        "directory": "set_2",
    },
    "set_3": {
        "name": "Cat Dataset",
        "module": cats,
        "directory": "set_3",
    },
}

def list_datasets() -> None:
    """ Print available datasets """

    print("Available datasets:\n")

    for dataset_id, dataset_info in DATASETS.items():
        print(f" - {dataset_id}: {dataset_info['name']}")

def get_dataset_path(dataset_id: str) -> Path:
    """
    Return the local cache for dataset
    """

    if dataset_id not in DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_id}. "
            f"Available datasets: {', '.join(DATASETS.keys())}"
        )
    
    return CACHE_DIR / DATASETS[dataset_id]["directory"]

def get_dataset(dataset_id: str) ->  Path:
    """Return a downloaded dataset path without downloading"""

    """NOTE: THIS IS WHAT SHOULD BE PASSED TO MODEL"""

    path = get_dataset_path(dataset_id)

    if not is_downloaded(dataset_id):
        raise FileNotFoundError(
            f"Dataset '{dataset_id}' is not downloaded. "
            f"Please download it first using download_dataset('{dataset_id}')"
        )

    return path

def is_downloaded(dataset_id: str) -> bool:
    """
    Check if the dataset is cached
    """

    path = get_dataset_path(dataset_id)

    return path.exists() and any(path.iterdir())

def download_dataset(dataset_id: str) -> None:
    """
    Download a dataset if not already cached

    Return the local path to the dataset
    """

    if dataset_id not in DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_id}. "
            f"Available datasets: {', '.join(DATASETS.keys())}"
        )

    dataset_info = DATASETS[dataset_id]
    destination = CACHE_DIR / dataset_info["directory"]

    # If already cached, don't redownload
    if is_downloaded(dataset_id):
        print(f"{dataset_info['name']} dataset already exists")
        print(f"Location: {destination}")
        return
    
    print(f"\nDownloading: {dataset_info['name']}")
    print(f"Destination: {destination}\n")

    dataset_info["module"].download(destination)

    print("\nDownload complete.")

    return destination

def _remove_readonly(func, path, _):
    """Allow shutil.rmtree to remove read-only files on Windows"""
    os.chmod(path, stat.S_IWRITE)
    func(path)

def remove_dataset(dataset_id: str) -> None:
    """
    Remove a dataset from the cache
    """

    import shutil

    if dataset_id not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset_id}'. "
            f"Available datasets: {', '.join(DATASETS)}"
        )

    dataset = DATASETS[dataset_id]
    destination = CACHE_DIR / dataset["directory"]

    if not destination.exists():
        print(f"{dataset['name']} is not currently downloaded.")
        return

    print(f"Removing: {destination}")

    shutil.rmtree(destination, onerror=_remove_readonly)

    print("Dataset removed from local cache.")

def remove_all_datasets() -> None:
    """
    Remove all datasets from the cache
    """

    import shutil

    if not CACHE_DIR.exists():
        print("No datasets are currently downloaded.")
        return

    print(f"Removing all datasets from: {CACHE_DIR}")

    shutil.rmtree(CACHE_DIR, onerror=_remove_readonly)

    print("All datasets removed from local cache.")

def remove_models() -> None:
    """
    Remove all models from the models directory
    """

    import shutil

    models_dir = Path(__file__).parent.parent / "models"

    if not models_dir.exists():
        print("No models are currently saved.")
        return

    print(f"Removing all models from: {models_dir}")

    shutil.rmtree(models_dir, onerror=_remove_readonly)

    print("All models removed from local cache.")
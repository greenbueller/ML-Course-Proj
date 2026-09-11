"""
Download the project datasets for yourself via command-line

Usage:

Download a dataset:
`python scripts/download_data.py set_1`

Download all datasets:
`python scripts/download_data.py all`

List available datasets:
`python scripts/download_data.py list`
"""

import sys
from pathlib import Path

# Get the project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import catloaf
from src.datasets import breeds
from src.datasets import cats

# Fetch from dataset manager
from src.datasets.manager import (
    DATASETS,
    download_dataset,
    list_datasets
)

def print_usage() -> None:
    print("Usage: python scripts/download_data.py <dataset_id>")
    print("Available datasets:")
    for key in DATASETS.keys():
        print(f"  - {key}")
    print("  - all")
    print("  - list")

def main() -> None:
    if len(sys.argv) != 2:
        print_usage()
        sys.exit(1)
    
    command = sys.argv[1]

    # List datasets

    if command == "list":
        list_datasets()
        return
    
    # Download all datasets

    if command == "all":
        print("Downloading all datasets...")

        for dataset_id in DATASETS:
            try:
                download_dataset(dataset_id)
                print()
            except Exception as e:
                print(f"\nERROR downloading {dataset_id}:")
                print(e)
                sys.exit(1)
        
        print("All datasets downloaded successfully.")
        return

        # Download a specific dataset

        if command not in DATASETS:
            print(f"Unknown dataset: {command}.")
            print_usage()
            sys.exit(1)
        
        try:
            path = download_dataset(command)

            print("\nDataset is ready.")
            print(f"Local path: {path}")

        except Exception as e:
            print("\nERROR:")
            print(e)
            sys.exit(1)

if __name__ == "__main__":
    main()
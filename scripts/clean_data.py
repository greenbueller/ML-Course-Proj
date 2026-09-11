"""
Command-line interface for removing locally cached datasets.

Examples
--------
Remove one dataset:

    python scripts/clean_data.py set_1

Remove all datasets:

    python scripts/clean_data.py all

List available datasets:

    python scripts/clean_data.py list
"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from src.datasets.manager import (  # noqa: E402
    DATASETS,
    remove_all_datasets,
    remove_dataset,
    list_datasets,
)


def print_usage() -> None:
    """Print command usage information."""

    print(
        """
Usage:

    python scripts/clean_data.py <dataset>

Commands:

    list
        Show available datasets.

    set_1
        Remove the Cat vs Cat-Loaf dataset.

    set_2
        Remove the Cat Breeds dataset.

    set_3
        Remove the Crawford Cat Dataset.

    all
        Remove all locally cached datasets.
"""
    )


def main() -> None:
    if len(sys.argv) != 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1]

    # ---------------------------------------------------------------
    # List datasets
    # ---------------------------------------------------------------

    if command == "list":
        list_datasets()
        return

    # ---------------------------------------------------------------
    # Remove all
    # ---------------------------------------------------------------

    if command == "all":
        confirmation = input(
            "Are you sure you want to delete ALL cached datasets? [y/N]: "
        )

        if confirmation.lower() != "y":
            print("Cancelled.")
            return

        remove_all_datasets()
        return

    # ---------------------------------------------------------------
    # Remove one dataset
    # ---------------------------------------------------------------

    if command not in DATASETS:
        print(f"Unknown dataset: {command}\n")
        print_usage()
        sys.exit(1)

    confirmation = input(
        f"Remove cached dataset '{command}'? [y/N]: "
    )

    if confirmation.lower() != "y":
        print("Cancelled.")
        return

    remove_dataset(command)


if __name__ == "__main__":
    main()
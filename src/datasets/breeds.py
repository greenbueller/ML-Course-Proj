from pathlib import Path
import subprocess

REPO = "https://github.com/atharvatras/cat-breeds-dataset.git"

COMMIT = "56a6905"

def download(destination: Path) -> None:
    """
    Download the Cat Breeds dataset from GitHub.

    Args:
        destination (Path): The destination directory to save the dataset.
    """
    destination.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        print(f"{destination} already exists. Skipping download.")
        return

    print("Cloning Cat Breeds dataset from GitHub...")

    subprocess.run(
        [
            "git",
            "clone",
            REPO,
            str(destination),
        ],
        check=True,
    )
    
    subprocess.run(
        [
            "git",
            "-C",
            str(destination),
            "checkout",
            COMMIT,
        ],
        check=True,
    )

    print(f"Dataset cloned to '{destination}'.")
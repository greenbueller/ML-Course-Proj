from pathlib import Path
import subprocess

REPO = "https://github.com/n-smith-public/Cat-Breeds-Loafing.git"

COMMIT = "55da393"

def download(destination: Path) -> None:
    """
    Download the Cat Breeds dataset from GitHub.

    Args:
        destination (Path): The destination directory to save the dataset.
    """
    if destination.exists() and any(destination.iterdir()):
        print(f"{destination} already exists. Skipping download.")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)

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
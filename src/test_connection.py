import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.manager import DATASETS, get_dataset


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp",
}


def count_images(directory: Path) -> int:
    return sum(
        1
        for file in directory.rglob("*")
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_dataset(dataset_id: str) -> bool:
    try:
        dataset_path = get_dataset(dataset_id)
    except FileNotFoundError as error:
        print(f"\nERROR: {error}")
        return False

    total_images = count_images(dataset_path)

    print(f"\n{dataset_id}: {dataset_path}")
    print(f"Total images: {total_images}")

    for directory in sorted(
        path for path in dataset_path.rglob("*") if path.is_dir()
    ):
        directory_image_count = count_images(directory)

        if directory_image_count:
            relative_path = directory.relative_to(dataset_path)
            print(f" - {relative_path}: {directory_image_count} images")

    if total_images == 0:
        print(f"Warning: No images found in dataset '{dataset_id}'.")
        return False

    return True


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python src/test_connection.py <dataset_id|all>")
        return 1

    requested_dataset = sys.argv[1]

    if requested_dataset == "all":
        dataset_ids = list(DATASETS)
    elif requested_dataset in DATASETS:
        dataset_ids = [requested_dataset]
    else:
        print(
            f"Unknown dataset: {requested_dataset}. "
            f"Available datasets: {', '.join(DATASETS)}"
        )
        return 1

    results = [
        validate_dataset(dataset_id)
        for dataset_id in dataset_ids
    ]

    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
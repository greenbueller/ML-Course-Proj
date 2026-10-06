import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.manager import CACHE_DIR, get_dataset

IMAGE_SIZE = (100, 100)
IMAGE_EXTENSIONS = { ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

DATASET_LAYOUTS = {
    "set_1": {
        "root": "images",
        "classes": {
            "loaf": "loaf",
            "cat": "not_loaf",
        },
    },
    "set_2": {
        "root": "",
        "classes": {
            "Loaf": "loaf",
            "Not loaf": "not_loaf",
        },
    },
}

NORMALISED_DIR = CACHE_DIR / "normalised"

def find_dir(parent: Path, name: str) -> Path:
    for child in parent.iterdir():
        if child.is_dir() and child.name == name:
            return child
    
    raise FileNotFoundError(f"Directory '{name}' not found in '{parent}'")

def normalise_dataset(dataset_id: str) -> Path:
    dataset_path = get_dataset(dataset_id)
    layout = DATASET_LAYOUTS[dataset_id]

    source_root = dataset_path / layout["root"]
    output_root = NORMALISED_DIR / dataset_id
    image_count = 0

    for source_class, output_class in layout["classes"].items():
        source_directory = find_dir(source_root, source_class)
        output_directory = output_root / output_class
        output_directory.mkdir(parents=True, exist_ok=True)

        for image_path in source_directory.rglob("*"):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            image = cv2.imread(str(image_path))

            if image is None:
                print(f"Skipping unreadable image: {image_path}")
                continue
            
            normalised_image = cv2.resize(
                image,
                IMAGE_SIZE,
                interpolation=cv2.INTER_AREA
            )
            
            output_path = output_directory / image_path.name
            success = cv2.imwrite(str(output_path), normalised_image)

            if not success:
                raise RuntimeError(f"Failed to write image: {output_path}")
            
            image_count += 1
    print(f"{dataset_id}: Normalised {image_count} images to {output_root}")
    return image_count

def main() -> None:
    total_images = 0

    for dataset_id in DATASET_LAYOUTS:
        total_images += normalise_dataset(dataset_id)
    
    print(f"\nNormalised {total_images} images across {len(DATASET_LAYOUTS)} datasets.")
    print(f"Output directory: {NORMALISED_DIR}")

if __name__ == "__main__":
    main()
from pathlib import Path
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.manager import CACHE_DIR, get_dataset

# ---------------------------------------------------------------
# Training arguments
# ---------------------------------------------------------------

args = {
    "batch_size": 32,
    "test_batch_size": 32,
    "epochs": 15,
    "lr": 0.01,
    "momentum": 0.75,
    "seed": 1,
    "log_interval": 10,
    "cuda": torch.cuda.is_available(),
}

IMAGE_SIZE = (100, 100)
NORMALISED_DIR = CACHE_DIR / "normalised"
MODEL_PATH = PROJECT_ROOT / "models" / "catloaf_cnn.pth"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# ---------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------

class CatLoafDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]]) -> None:
        self.samples = samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        image_path, label = self.samples[idx]

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMAGE_SIZE)

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # Convert to (C, H, W)

        image_tensor = torch.tensor(image, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor

def collect_samples() -> list[tuple[Path, int]]:
    """
    Load normalised images

    Label 0: Not Loafing
    Label 1: Loafing
    """

    samples = []

    for dataset_id in ["set_1", "set_2"]:
        dataset_root = NORMALISED_DIR / dataset_id

        class_directories = {
            "not_loaf": 0,
            "loaf": 1,
        }

        for class_name, label in class_directories.items():
            class_dir = dataset_root / class_name

            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Missing directory: {class_dir}\n"
                    "Run `python src/normalise.py` first."
                )
            
            for image_path in class_dir.rglob("*"):
                if (
                    image_path.is_file()
                    and image_path.suffix.lower() in IMAGE_EXTENSIONS
                ):
                    samples.append((image_path, label))
            
    if not samples:
        raise RuntimeError("No normalised images were found.")
        
    return samples

def create_data_loaders():
    samples = collect_samples()

    labels = [label for _, label in samples]
    indices = list(range(len(samples)))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=args["seed"],
        stratify=labels
    )

    dataset = CatLoafDataset(samples)

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    kwargs = {
        "num_workers": 0,
        "pin_memory": args["cuda"],
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        shuffle=True,
        **kwargs
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args["test_batch_size"],
        shuffle=False,
        **kwargs
    )

    print(f"Total images: {len(samples)}")
    print(f"Training images: {len(train_indices)}")
    print(f"Testing images: {len(test_indices)}")

    return train_loader, test_loader

# ---------------------------------------------------------------
# Model
# ---------------------------------------------------------------

class NetCNN(nn.Module):
    def __init__(self):
        super(NetCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=5,
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Input: 100x100
        # conv1 -> 96x96
        # pool  -> 48x48
        # conv2 -> 44x44
        # pool  -> 22x22
        self.fc1 = nn.Linear(64 * 22 * 22, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
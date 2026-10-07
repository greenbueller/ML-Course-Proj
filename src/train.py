# ---------------------------------------------------------------
# Import statements and setup, do not change.
# ---------------------------------------------------------------

from pathlib import Path
from collections import Counter
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.manager import CACHE_DIR, get_dataset

# ---------------------------------------------------------------
# Training arguments
# ---------------------------------------------------------------

# CHANGE AS NEEDED
"""
args:
    batch_size: Number of samples per batch during training. Change if encounter a memory error.
    test_batch_size: Number of samples per batch during testing. Change if encounter a memory error.
    epochs: Number of complete passes through the training dataset.
    lr: Learning rate for the optimizer.
    momentum: Momentum factor for certain optimizers.
    seed: Random seed for reproducibility.
    log_interval: How often to log training progress (in batches).
    cuda: Whether to use CUDA for training. Do not change.
"""
args = {
    "batch_size": 128,
    "test_batch_size": 128,
    "epochs": 30,
    "lr": 0.0003,
    "momentum": 0.00,
    "seed": 1,
    "log_interval": 10,
    "cuda": torch.cuda.is_available(),
}

# DO NOT CHANGE
IMAGE_SIZE = (100, 100)
NORMALISED_DIR = CACHE_DIR / "normalised"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
# CHANGE FINAL PART ONLY
MODEL_PATH = PROJECT_ROOT / "models" / "catloaf.pth"

# Passes in the seed to all random number generators for reproducibility
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# ---------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------

class CatLoafDataset(Dataset):
    # Set training=true when calling if want to run the image augmentation
    def __init__(self, samples, training=False):
        self.samples = samples
        self.training = training
    
    # Number of samples
    def __len__(self) -> int:
        return len(self.samples)
    
    # Get a sample
    def __getitem__(self, idx: int):
        image_path, label = self.samples[idx]

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        
        # Convert BGR to RGB and resize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMAGE_SIZE)

        # Image augmentation
        if self.training:
            if random.random() < 0.5:
                image = cv2.flip(image, 1) # Horizontal flip
            
            if random.random() < 0.3:
                brightness = random.uniform(0.9, 1.1) # Change brightness slightly
                image = np.clip(
                    image.astype(np.float32) * brightness,
                    0,
                    255,
                ).astype(np.uint8)

        # Normalize and convert to tensor
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        image_tensor = torch.tensor(image, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor

def collect_samples(dataset_ids):
    """
    Load normalised images

    Label 0: Not Loafing
    Label 1: Loafing
    """

    samples = []

    # Takes passed in dataset id
    for dataset_id in dataset_ids:
        dataset_root = NORMALISED_DIR / dataset_id

        # Gets class names/labels
        for class_name, label in {
            "not_loaf": 0,
            "loaf": 1,
        }.items():
            class_dir = dataset_root / class_name

            # Must have the classes
            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Missing directory: {class_dir}\n"
                    "Run `python src/normalise.py` first."
                )
            
            # If the image is valid, add it to the samples list
            for image_path in class_dir.rglob("*"):
                if (
                    image_path.is_file()
                    and image_path.suffix.lower() in IMAGE_EXTENSIONS
                ):
                    samples.append((image_path, label))
            
    if not samples:
        raise RuntimeError("No normalised images were found.")
        
    return samples

def balance_samples(samples, seed=1):
    """
    Get a fair balance of loaf and non-loaf samples.

    Mainly used because set_2 is highly imbalanced with more non-loaf images than loaf.
    """
    loaf_samples = [
        sample for sample in samples
        if sample[1] == 1
    ]

    not_loaf_samples = [
        sample for sample in samples
        if sample[1] == 0
    ]

    rng = random.Random(seed)
    not_loaf_samples = rng.sample(
        not_loaf_samples,
        len(loaf_samples)
    )

    return loaf_samples + not_loaf_samples

def create_data_loaders():
    """
    Loads in the datasets. Split into set_1 and set_2, as only train on set 1 but test on set 1 and 2.
    """
    set_1_samples = collect_samples(["set_1"])
    set_2_samples = collect_samples(["set_2"])
    set_2_samples = balance_samples(set_2_samples, seed=args["seed"])

    labels = [label for _, label in set_1_samples]
    indices = list(range(len(set_1_samples)))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=args["seed"],
        stratify=labels
    )

    train_samples = [set_1_samples[index] for index in train_indices]
    set_1_test_samples = [set_1_samples[index] for index in test_indices]

    train_dataset = CatLoafDataset(train_samples, training=False)
    set_1_test_dataset = CatLoafDataset(
        set_1_test_samples,
        training=False,
    )
    set_2_dataset = CatLoafDataset(set_2_samples, training=False)

    train_labels = [labels[index] for index in train_indices]

    class_counts = Counter(train_labels)

    class_weights = torch.tensor(
        [
            len(train_labels) / (2 * class_counts[0]),
            len(train_labels) / (2 * class_counts[1]),
        ],
        dtype=torch.float32
    )

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

    set_1_test_loader = DataLoader(
        set_1_test_dataset,
        batch_size=args["test_batch_size"],
        shuffle=False,
        **kwargs
    )

    set_2_loader = DataLoader(
        set_2_dataset,
        batch_size=args["test_batch_size"],
        shuffle=False,
        **kwargs
    )

    return (
        train_loader,
        set_1_test_loader,
        set_2_loader,
        class_weights
    )

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
        # adaptive_pool -> 8x8
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 2)
        # Add batch normalisation
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        # Dropout layer, reduces the chance of overfitting by adding chance to reduce lr
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

# ---------------------------------------------------------------
# Training
# ---------------------------------------------------------------

def train(epoch, model, train_loader, optimiser, device, class_weights):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        optimiser.zero_grad()

        output = model(data)

        loss = F.nll_loss(output, target, weight=class_weights)

        loss.backward()
        optimiser.step()

        if batch_idx % args["log_interval"] == 0:
            processed = batch_idx * len(data)
            total = len(train_loader.dataset)
            percent = 100.0 * batch_idx / len(train_loader)

            print(
                f"Train Epoch: {epoch} / {args['epochs']} "
                f"[{processed}/{total} ({percent:.0f}%)]"
                f" Loss: {loss.item():.6f}"
            )

def test(model, test_loader, device, dataset_name):
    model.eval()

    correct = 0    
    actual = []
    predicted = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            pred = output.argmax(dim=1)

            correct += pred.eq(target).sum().item()

            actual.extend(target.cpu().tolist())
            predicted.extend(pred.cpu().tolist())
    
    # test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    loaf_f1 = f1_score(
        actual,
        predicted,
        pos_label=1,
        zero_division=0
    )

    """
    args:
        - dataset_name: Name of the dataset being evaluated.
        - accuracy: Overall accuracy of the model on the dataset.
        - loaf_f1: F1 score for the "loaf" class.
            - The F1 score is part of confusion matrices, where a higher F1 score means there was better performance.
        - classification_report: Detailed classification report including precision, recall, and F1 score for each class.
    """
    print(
        f"\n{dataset_name} | "
        f"Accuracy: {accuracy:.2f}%, "
        f"Loaf F1 Score: {loaf_f1:.4f}"
        f"\n{classification_report(actual, predicted, target_names=["not_loaf", "loaf"], zero_division=0)}"
    )

    return accuracy, loaf_f1

def main():
    # Input the seed from arguments
    seed_everything(args["seed"])

    # If have CUDA enabled, use it. Otherwise, use CPU
    device = torch.device( "cuda" if args["cuda"] else "cpu")

    # Debugging statements to ensure the correct device is being used and the PyTorch version is compatible
    ## If these are not what you expect, check your PyTorch installation and CUDA setup.
    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {device}")

    # Load data
    (
        train_loader,
        set_1_test_loader,
        set_2_loader,
        class_weights
    ) = create_data_loaders()

    # Send the class weights
    class_weights = class_weights.to(device)

    # Inspect the first batch of training data
    for images, labels in train_loader:
        print("Image batch dimensions:", images.shape)
        print("Label batch dimensions:", labels.shape)
        break
    
    # Create the model
    model = NetCNN().to(device)
    print(model)

    # Previously used SGD, then Adam, now trying AdamW. Feel free to change if you want to experiment
    """
        Per https://docs.pytorch.org/docs/main/optim.html

        Available Optimisers:
        - Adaleta
        - Adafactor
        - Adagrad
        - Adam
        - AdamW, weight decay does not accumulate in momentum nor variance
        - SparseAdam
        - Adamax
        - ASGD
        - LBFGS
        - Muon
        - NAdam
        - RAdam
        - RMSprop
        - Rprop
        - SGD, what we used in HW3
    """
    optimiser = optim.AdamW(
        model.parameters(),
        lr=args["lr"],
        weight_decay=0.0001,
    )

    # Introduces a scheduler that reduces learning rate if the f1 score does not change for patience epochs.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr = 1e-6,
    )

    # Time out for training
    best_f1 = 0.0
    patience = 5
    epoch_no_improvement = 0

    # Create the model directory if needed
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Training
    ## For the number of epochs
    for epoch in range(1, args["epochs"] + 1):
        # Train the model
        train(
            epoch, 
            model, 
            train_loader, 
            optimiser, 
            device, 
            class_weights
        )
        
        # Get the accuracy and f1 score for the validation set
        set_1_accuracy, set_1_f1 = test(
            model, 
            set_1_test_loader, 
            device, 
            "Set 1 Validation"
        )
        
        # Tell scheduler the f1 score for this epoch, so it can adjust the learning rate if needed
        scheduler.step(set_1_f1)

        # Get the current learning rate from the optimiser
        current_lr = optimiser.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}:"
            f"accuracy={set_1_accuracy:.2f}%, "
            f"loaf_f1={set_1_f1:.4f}, "
            f"lr = {current_lr:.6f}"
        )

        # If we have a better f1 score than before
        if set_1_f1 > best_f1:
            # Save it
            best_f1 = set_1_f1
            # Reset counter
            epoch_no_improvement = 0

            print(f"New best loaf F1 score: {best_f1:.4f}. Saving model...")

            # Save the model
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_size": IMAGE_SIZE,
                    "class_names": {
                        0: "not_loaf",
                        1: "loaf",
                    },
                },
                MODEL_PATH
            )
        # Otherwise, increase the counter
        else:
            epoch_no_improvement += 1
            
        # If we hit the patience, we end training early to not waste time
        if epoch_no_improvement >= patience:
            print(
                f"No improvement in F1 score for {patience} epochs."
                " Stopping training early."
            )
            break

    # Run against test datasets
    print(f"Loading best model for final evaluation: {MODEL_PATH}")

    # Load the best model
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Test on the set 1 test set
    print("\nFinal evaluation on Set 1 Test Set:")
    test(
        model,
        set_1_test_loader,
        device,
        "Set 1 Test Set"
    )

    # Test on sample from set 2
    print("\nFinal evaluation on Set 2:")
    test(
        model,
        set_2_loader,
        device,
        "Set 2"
    )

    print(f"\nBest model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
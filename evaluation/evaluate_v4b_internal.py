from pathlib import Path
import sys

PROJECT_DIR = Path(r"D:\SignalScope")
sys.path.insert(0, str(PROJECT_DIR))

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from model.model import load_model


# ============================================================
# Configuration
# ============================================================

DATA_DIR = (
    PROJECT_DIR
    / "data"
    / "dataset_v4a"
    / "test"
)

MODEL_PATH = (
    PROJECT_DIR
    / "model"
    / "signalscope_resnet18_v4b.pth"
)

IMAGE_SIZE = 224
BATCH_SIZE = 32


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# Transform
# Same deterministic preprocessing used for V4-A testing
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# Dataset
# ============================================================

dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print("\nClass mapping:")
print(dataset.class_to_idx)

print("\nTest images:", len(dataset))


# ============================================================
# Load V4-B
# ============================================================

print("\nLoading V4-B model...")

model = load_model(
    MODEL_PATH,
    device
)

print("Model loaded successfully.")


# ============================================================
# Evaluation
# ============================================================

all_labels = []
all_predictions = []
all_real_probabilities = []

model.eval()

with torch.no_grad():

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        real_probabilities = probabilities[:, 1]

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_real_probabilities.extend(
            real_probabilities.cpu().numpy()
        )


# ============================================================
# Metrics
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    zero_division=0
)

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

roc_auc = roc_auc_score(
    all_labels,
    all_real_probabilities
)

cm = confusion_matrix(
    all_labels,
    all_predictions
)


# ============================================================
# Results
# ============================================================

print("\n" + "=" * 70)
print("V4-B — INTERNAL V4-A TEST SET")
print("=" * 70)

print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"Macro-F1:  {macro_f1 * 100:.2f}%")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=["AI", "Real"],
        zero_division=0
    )
)

print("=" * 70)
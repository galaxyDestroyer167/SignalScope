from pathlib import Path
import sys

PROJECT_DIR = Path(r"D:\SignalScope")
sys.path.insert(0, str(PROJECT_DIR))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import accuracy_score, coverage_error

from model.model import load_model


# ============================================================
# SETTINGS
# ============================================================

PROJECT_DIR = Path(r"D:\SignalScope")

MODEL_PATH = PROJECT_DIR / "model" / "signalscope_resnet18_v4b.pth"
VAL_DIR = PROJECT_DIR / "data" / "dataset_v4a" / "val"

BATCH_SIZE = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASET
# ============================================================

dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


print("=" * 70)
print("V4-B ABSTENTION / UNCERTAINTY EXPERIMENT")
print("=" * 70)

print()
print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print()
print("Class mapping:")
print(dataset.class_to_idx)

print()
print("Validation images:", len(dataset))


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading V4-B model...")

model = load_model(
    MODEL_PATH,
    DEVICE
)

print("Model loaded successfully.")


# ============================================================
# COLLECT PREDICTIONS
# ============================================================

all_labels = []
all_confidences = []
all_predictions = []

model.eval()

with torch.no_grad():

    for images, labels in loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = F.softmax(outputs, dim=1)

        confidence, predictions = torch.max(
            probabilities,
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_confidences.extend(
            confidence.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


# ============================================================
# BASELINE
# ============================================================

all_labels = torch.tensor(all_labels).numpy()
all_confidences = torch.tensor(all_confidences).numpy()
all_predictions = torch.tensor(all_predictions).numpy()

baseline_accuracy = accuracy_score(
    all_labels,
    all_predictions
)


print()
print("=" * 70)
print("BASELINE")
print("=" * 70)

print(
    f"Accuracy without abstention: "
    f"{baseline_accuracy * 100:.2f}%"
)


# ============================================================
# ABSTENTION EXPERIMENT
# ============================================================

print()
print("=" * 70)
print("ABSTENTION RESULTS")
print("=" * 70)

print()

print(
    f"{'Threshold':<12}"
    f"{'Coverage':<12}"
    f"{'Abstain':<12}"
    f"{'Accuracy':<12}"
)

print("-" * 48)


thresholds = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]


for threshold in thresholds:

    accepted = all_confidences >= threshold

    accepted_count = accepted.sum()

    total_count = len(all_labels)

    coverage = accepted_count / total_count

    abstained = total_count - accepted_count

    if accepted_count > 0:

        accuracy = accuracy_score(
            all_labels[accepted],
            all_predictions[accepted]
        )

    else:

        accuracy = 0.0


    print(
        f"{threshold:<12.2f}"
        f"{coverage * 100:<12.2f}"
        f"{abstained:<12}"
        f"{accuracy * 100:<12.2f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

output_file = (
    PROJECT_DIR /
    "evaluation" /
    "abstention_v4b_results.txt"
)

with open(output_file, "w", encoding="utf-8") as f:

    f.write("V4-B ABSTENTION / UNCERTAINTY EXPERIMENT\n")
    f.write("=" * 60 + "\n\n")

    f.write(
        f"Baseline accuracy: "
        f"{baseline_accuracy * 100:.2f}%\n\n"
    )

    f.write(
        f"{'Threshold':<12}"
        f"{'Coverage':<12}"
        f"{'Abstain':<12}"
        f"{'Accuracy':<12}\n"
    )

    f.write("-" * 48 + "\n")

    for threshold in thresholds:

        accepted = all_confidences >= threshold

        accepted_count = accepted.sum()

        total_count = len(all_labels)

        coverage = accepted_count / total_count

        abstained = total_count - accepted_count

        if accepted_count > 0:

            accuracy = accuracy_score(
                all_labels[accepted],
                all_predictions[accepted]
            )

        else:

            accuracy = 0.0

        f.write(
            f"{threshold:<12.2f}"
            f"{coverage * 100:<12.2f}"
            f"{abstained:<12}"
            f"{accuracy * 100:<12.2f}\n"
        )


print()
print("Results saved to:")
print(output_file)

print()
print("=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)
from pathlib import Path
import sys

PROJECT_DIR = Path(r"D:\SignalScope")
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from model.model import load_model


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = (
    PROJECT_DIR /
    "model" /
    "signalscope_resnet18_v4b.pth"
)

VAL_DIR = (
    PROJECT_DIR /
    "data" /
    "dataset_v4a" /
    "val"
)

BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("V4-B CONFIDENCE / CALIBRATION ANALYSIS")
print("=" * 70)

print()
print("Device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

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
all_predictions = []
all_confidences = []

model.eval()

with torch.no_grad():

    for images, labels in loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = F.softmax(
            outputs,
            dim=1
        )

        confidence, predictions = torch.max(
            probabilities,
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_confidences.extend(
            confidence.cpu().numpy()
        )


labels = np.array(all_labels)
predictions = np.array(all_predictions)
confidences = np.array(all_confidences)


# ============================================================
# BASIC ANALYSIS
# ============================================================

correct = predictions == labels
incorrect = ~correct

accuracy = accuracy_score(
    labels,
    predictions
)

print()
print("=" * 70)
print("BASIC RESULTS")
print("=" * 70)

print()
print(
    f"Validation accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Correct predictions: "
    f"{correct.sum()}"
)

print(
    f"Incorrect predictions: "
    f"{incorrect.sum()}"
)

print()
print(
    f"Average confidence - correct: "
    f"{confidences[correct].mean() * 100:.2f}%"
)

print(
    f"Average confidence - incorrect: "
    f"{confidences[incorrect].mean() * 100:.2f}%"
)


# ============================================================
# CONFIDENCE BINS
# ============================================================

print()
print("=" * 70)
print("CONFIDENCE VS ACCURACY")
print("=" * 70)

bins = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00
]

print()
print(
    f"{'Confidence':<18}"
    f"{'Images':<10}"
    f"{'Accuracy':<12}"
)

print("-" * 40)

bin_results = []

for low, high in zip(
    bins[:-1],
    bins[1:]
):

    mask = (
        (confidences >= low) &
        (confidences < high)
    )

    count = mask.sum()

    if count > 0:

        bin_accuracy = (
            predictions[mask] ==
            labels[mask]
        ).mean()

    else:

        bin_accuracy = 0.0

    bin_results.append(
        (
            low,
            high,
            count,
            bin_accuracy
        )
    )

    print(
        f"{low * 100:.0f}-{high * 100:.0f}%"
        f"{'':<11}"
        f"{count:<10}"
        f"{bin_accuracy * 100:.2f}%"
    )


# ============================================================
# SELECTIVE ACCURACY
# ============================================================

print()
print("=" * 70)
print("SELECTIVE ACCURACY")
print("=" * 70)

thresholds = np.arange(
    0.50,
    1.00,
    0.05
)

selective_results = []

print()
print(
    f"{'Threshold':<12}"
    f"{'Coverage':<12}"
    f"{'Accuracy':<12}"
)

print("-" * 36)

for threshold in thresholds:

    accepted = (
        confidences >= threshold
    )

    coverage = accepted.mean()

    if accepted.sum() > 0:

        selective_accuracy = (
            predictions[accepted] ==
            labels[accepted]
        ).mean()

    else:

        selective_accuracy = 0.0

    selective_results.append(
        (
            threshold,
            coverage,
            selective_accuracy
        )
    )

    print(
        f"{threshold * 100:<12.0f}"
        f"{coverage * 100:<12.2f}"
        f"{selective_accuracy * 100:.2f}%"
    )


# ============================================================
# RELIABILITY CURVE
# ============================================================

reliability_confidence = []
reliability_accuracy = []
reliability_count = []

for low, high in zip(
    bins[:-1],
    bins[1:]
):

    mask = (
        (confidences >= low) &
        (confidences < high)
    )

    if mask.sum() == 0:
        continue

    reliability_confidence.append(
        confidences[mask].mean()
    )

    reliability_accuracy.append(
        correct[mask].mean()
    )

    reliability_count.append(
        mask.sum()
    )


plt.figure(figsize=(8, 6))

plt.plot(
    reliability_confidence,
    reliability_accuracy,
    marker="o",
    label="V4-B"
)

plt.plot(
    [0.5, 1.0],
    [0.5, 1.0],
    linestyle="--",
    label="Perfect calibration"
)

plt.xlabel("Average Confidence")
plt.ylabel("Accuracy")
plt.title(
    "V4-B Reliability Curve"
)

plt.xlim(0.5, 1.0)
plt.ylim(0.5, 1.0)

plt.grid(True)
plt.legend()

reliability_path = (
    PROJECT_DIR /
    "evaluation" /
    "reliability_curve_v4b.png"
)

plt.savefig(
    reliability_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SELECTIVE ACCURACY CURVE
# ============================================================

coverage_values = [
    x[1]
    for x in selective_results
]

accuracy_values = [
    x[2]
    for x in selective_results
]


plt.figure(figsize=(8, 6))

plt.plot(
    coverage_values,
    accuracy_values,
    marker="o"
)

plt.xlabel("Coverage")
plt.ylabel("Accuracy")

plt.title(
    "V4-B Selective Accuracy Curve"
)

plt.grid(True)

selective_path = (
    PROJECT_DIR /
    "evaluation" /
    "selective_accuracy_v4b.png"
)

plt.savefig(
    selective_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_path = (
    PROJECT_DIR /
    "evaluation" /
    "calibration_analysis_v4b.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "V4-B CONFIDENCE / CALIBRATION ANALYSIS\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        f"Validation accuracy: "
        f"{accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Correct: {correct.sum()}\n"
    )

    f.write(
        f"Incorrect: {incorrect.sum()}\n\n"
    )

    f.write(
        f"Average confidence - correct: "
        f"{confidences[correct].mean() * 100:.2f}%\n"
    )

    f.write(
        f"Average confidence - incorrect: "
        f"{confidences[incorrect].mean() * 100:.2f}%\n\n"
    )

    f.write(
        "CONFIDENCE VS ACCURACY\n"
    )

    f.write("-" * 40 + "\n")

    for low, high, count, bin_accuracy in bin_results:

        f.write(
            f"{low * 100:.0f}-{high * 100:.0f}%: "
            f"{count} images, "
            f"{bin_accuracy * 100:.2f}% accuracy\n"
        )

    f.write("\n")
    f.write("SELECTIVE ACCURACY\n")
    f.write("-" * 40 + "\n")

    for threshold, coverage, selective_accuracy in selective_results:

        f.write(
            f"{threshold * 100:.0f}% threshold: "
            f"{coverage * 100:.2f}% coverage, "
            f"{selective_accuracy * 100:.2f}% accuracy\n"
        )


print()
print("=" * 70)
print("FILES CREATED")
print("=" * 70)

print()
print(reliability_path)
print(selective_path)
print(summary_path)

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
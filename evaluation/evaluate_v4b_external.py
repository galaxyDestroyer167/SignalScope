from pathlib import Path
import sys

PROJECT_DIR = Path(r"D:\SignalScope")
sys.path.insert(0, str(PROJECT_DIR))

import torch
from torchvision import transforms
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from model.model import load_model


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = (
    PROJECT_DIR
    / "model"
    / "signalscope_resnet18_v4b.pth"
)

AI_FOLDER = (
    PROJECT_DIR
    / "data"
    / "external_test"
    / "ai"
)

REAL_FOLDER = (
    PROJECT_DIR
    / "data"
    / "external_test"
    / "real"
)

IMAGE_SIZE = 224


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
# Same preprocessing as V4-A
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
# Load model
# ============================================================

print("\nLoading V4-B model...")

model = load_model(
    MODEL_PATH,
    device
)

print("Model loaded successfully.")


# ============================================================
# Collect images
# ============================================================

ai_files = sorted(
    AI_FOLDER.glob("*.png")
)

real_files = sorted(
    REAL_FOLDER.glob("*.jpg")
)

files = ai_files + real_files

labels = (
    [0] * len(ai_files)
    +
    [1] * len(real_files)
)

print("\nImages:")
print("AI:", len(ai_files))
print("Real:", len(real_files))
print("Total:", len(files))


# ============================================================
# Predictions
# ============================================================

all_predictions = []
all_real_probabilities = []

model.eval()

with torch.no_grad():

    for image_path in files:

        image = Image.open(
            image_path
        ).convert("RGB")

        input_tensor = transform(
            image
        ).unsqueeze(0).to(device)

        output = model(
            input_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

        prediction = torch.argmax(
            probabilities
        ).item()

        real_probability = (
            probabilities[1].item()
        )

        all_predictions.append(
            prediction
        )

        all_real_probabilities.append(
            real_probability
        )

        label = (
            "AI"
            if prediction == 0
            else "Real"
        )

        confidence = (
            probabilities[prediction].item()
            * 100
        )

        print(
            f"{image_path.name:15s} | "
            f"Actual: "
            f"{'AI' if labels[len(all_predictions)-1] == 0 else 'Real':5s} | "
            f"Predicted: {label:5s} | "
            f"Confidence: {confidence:6.2f}%"
        )


# ============================================================
# Metrics
# ============================================================

accuracy = accuracy_score(
    labels,
    all_predictions
)

precision = precision_score(
    labels,
    all_predictions,
    zero_division=0
)

recall = recall_score(
    labels,
    all_predictions,
    zero_division=0
)

macro_f1 = f1_score(
    labels,
    all_predictions,
    average="macro",
    zero_division=0
)

roc_auc = roc_auc_score(
    labels,
    all_real_probabilities
)

cm = confusion_matrix(
    labels,
    all_predictions
)


# ============================================================
# Final results
# ============================================================

print("\n" + "=" * 70)
print("V4-B — EXTERNAL 60 IMAGE TEST")
print("=" * 70)

print(
    f"Accuracy:  {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall:    {recall * 100:.2f}%"
)

print(
    f"Macro-F1:  {macro_f1 * 100:.2f}%"
)

print(
    f"ROC-AUC:   {roc_auc:.4f}"
)

print("\nConfusion Matrix:")
print(cm)

print("=" * 70)
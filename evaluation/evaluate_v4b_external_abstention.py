from pathlib import Path
import sys

PROJECT_DIR = Path(r"D:\SignalScope")
sys.path.insert(0, str(PROJECT_DIR))

import torch

from model.predict import predict_image


# ============================================================
# SETTINGS
# ============================================================

EXTERNAL_AI_DIR = PROJECT_DIR / "data" / "external_test" / "ai"
EXTERNAL_REAL_DIR = PROJECT_DIR / "data" / "external_test" / "real"

ABSTENTION_THRESHOLD = 85.0


# ============================================================
# TEST FUNCTION
# ============================================================

def test_folder(folder, true_label):

    results = []

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    }

    image_files = sorted(
        [
            p for p in folder.iterdir()
            if p.suffix.lower() in image_extensions
        ]
    )

    for image_path in image_files:

        result = predict_image(
            str(image_path)
        )

        label = result["label"]
        confidence = result["confidence"]

        if confidence < ABSTENTION_THRESHOLD:

            decision = "UNCERTAIN"

        else:

            decision = label

        results.append({
            "file": image_path.name,
            "true": true_label,
            "label": label,
            "confidence": confidence,
            "decision": decision
        })

    return results


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("V4-B EXTERNAL TEST WITH ABSTENTION")
print("=" * 70)

print()
print("Device:", "cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()
print(
    f"Abstention threshold: "
    f"{ABSTENTION_THRESHOLD:.0f}%"
)


# ============================================================
# RUN TEST
# ============================================================

print()
print("Testing AI images...")

ai_results = test_folder(
    EXTERNAL_AI_DIR,
    "AI"
)

print("Testing Real images...")

real_results = test_folder(
    EXTERNAL_REAL_DIR,
    "Real"
)

results = ai_results + real_results


# ============================================================
# DISPLAY INDIVIDUAL RESULTS
# ============================================================

print()
print("=" * 70)
print("INDIVIDUAL RESULTS")
print("=" * 70)

for r in results:

    print(
        f"{r['file']:<15} "
        f"True={r['true']:<5} "
        f"Model={r['label']:<5} "
        f"Confidence={r['confidence']:>6.2f}% "
        f"Decision={r['decision']}"
    )


# ============================================================
# SUMMARY
# ============================================================

total = len(results)

uncertain = sum(
    r["decision"] == "UNCERTAIN"
    for r in results
)

decided = total - uncertain

correct_decisions = sum(
    r["decision"] == r["true"]
    for r in results
    if r["decision"] != "UNCERTAIN"
)


print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print()
print(f"Total images:       {total}")
print(f"Automatic decisions:{decided}")
print(f"Uncertain:          {uncertain}")

if total > 0:

    print(
        f"Coverage:           "
        f"{decided / total * 100:.2f}%"
    )

if decided > 0:

    print(
        f"Accuracy on decided:"
        f" {correct_decisions / decided * 100:.2f}%"
    )


# ============================================================
# CLASS-WISE SUMMARY
# ============================================================

for true_label, class_results in [
    ("AI", ai_results),
    ("Real", real_results)
]:

    class_total = len(class_results)

    class_uncertain = sum(
        r["decision"] == "UNCERTAIN"
        for r in class_results
    )

    class_decided = class_total - class_uncertain

    class_correct = sum(
        r["decision"] == true_label
        for r in class_results
        if r["decision"] != "UNCERTAIN"
    )

    print()
    print(f"{true_label} images:")
    print(
        f"  Total:             {class_total}"
    )
    print(
        f"  Automatic:         {class_decided}"
    )
    print(
        f"  Uncertain:         {class_uncertain}"
    )

    if class_decided > 0:

        print(
            f"  Accuracy decided:  "
            f"{class_correct / class_decided * 100:.2f}%"
        )


print()
print("=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)
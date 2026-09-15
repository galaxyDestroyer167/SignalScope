from pathlib import Path
import sys
import io

PROJECT_DIR = Path(r"D:\SignalScope")
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
from PIL import Image, ImageFilter

from model.predict import predict_image


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "V4-B"

SOURCE_AI_DIR = (
    PROJECT_DIR /
    "data" /
    "external_test" /
    "ai"
)

SOURCE_REAL_DIR = (
    PROJECT_DIR /
    "data" /
    "external_test" /
    "real"
)

OUTPUT_DIR = (
    PROJECT_DIR /
    "evaluation" /
    "robustness_variants_v4b"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TRANSFORM FUNCTIONS
# ============================================================

def jpeg_compress(image, quality):

    buffer = io.BytesIO()

    image = image.convert("RGB")

    image.save(
        buffer,
        format="JPEG",
        quality=quality
    )

    buffer.seek(0)

    return Image.open(buffer).convert("RGB")


def resize_image(image, scale):

    width, height = image.size

    new_width = max(
        32,
        int(width * scale)
    )

    new_height = max(
        32,
        int(height * scale)
    )

    return image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )


def blur_image(image):

    return image.filter(
        ImageFilter.GaussianBlur(
            radius=1.5
        )
    )


def add_noise(image):

    image_array = np.array(
        image.convert("RGB")
    ).astype(np.float32)

    noise = np.random.normal(
        0,
        8,
        image_array.shape
    )

    noisy = image_array + noise

    noisy = np.clip(
        noisy,
        0,
        255
    ).astype(np.uint8)

    return Image.fromarray(
        noisy
    )


def crop_image(image):

    width, height = image.size

    crop_width = int(
        width * 0.8
    )

    crop_height = int(
        height * 0.8
    )

    left = (
        width - crop_width
    ) // 2

    top = (
        height - crop_height
    ) // 2

    right = left + crop_width
    bottom = top + crop_height

    return image.crop(
        (
            left,
            top,
            right,
            bottom
        )
    )


# ============================================================
# VARIANTS
# ============================================================

def create_variants(image):

    return {

        "original": image,

        "jpeg_95": jpeg_compress(
            image,
            95
        ),

        "jpeg_75": jpeg_compress(
            image,
            75
        ),

        "jpeg_50": jpeg_compress(
            image,
            50
        ),

        "resize_75": resize_image(
            image,
            0.75
        ),

        "resize_50": resize_image(
            image,
            0.50
        ),

        "blur": blur_image(
            image
        ),

        "noise": add_noise(
            image
        ),

        "crop": crop_image(
            image
        )
    }


# ============================================================
# FIND IMAGES
# ============================================================

def get_images(folder):

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    }

    return sorted(
        [
            p
            for p in folder.iterdir()
            if p.suffix.lower()
            in extensions
        ]
    )


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("V4-B REAL-WORLD ROBUSTNESS TEST")
print("=" * 70)

print()
print("Model:", MODEL_NAME)

print()
print("AI source:", SOURCE_AI_DIR)
print("Real source:", SOURCE_REAL_DIR)

print()
print("Creating controlled image variants...")


# ============================================================
# CREATE VARIANT DATASET
# ============================================================

source_sets = [
    ("AI", SOURCE_AI_DIR),
    ("Real", SOURCE_REAL_DIR)
]

all_variant_files = []


for true_label, source_dir in source_sets:

    images = get_images(
        source_dir
    )

    label_output_dir = (
        OUTPUT_DIR /
        true_label
    )

    label_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for image_path in images:

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            variants = create_variants(
                image
            )

            base_name = (
                image_path.stem
            )

            for variant_name, variant in variants.items():

                output_path = (
                    label_output_dir /
                    f"{base_name}_{variant_name}.jpg"
                )

                variant.convert(
                    "RGB"
                ).save(
                    output_path,
                    "JPEG",
                    quality=95
                )

                all_variant_files.append(
                    (
                        true_label,
                        variant_name,
                        output_path
                    )
                )

        except Exception as e:

            print(
                f"ERROR processing "
                f"{image_path.name}: {e}"
            )


print()
print(
    "Total generated variants:",
    len(all_variant_files)
)


# ============================================================
# LOAD MODEL ON FIRST PREDICTION
# ============================================================

print()
print("Loading V4-B model...")

# predict_image loads the model internally.
# We simply perform predictions using the existing
# SignalScope prediction pipeline.

print("Model ready.")


# ============================================================
# RUN PREDICTIONS
# ============================================================

results = []

print()
print("=" * 70)
print("RUNNING PREDICTIONS")
print("=" * 70)

for index, (
    true_label,
    variant_name,
    image_path
) in enumerate(
    all_variant_files,
    start=1
):

    try:

        prediction = predict_image(
            str(image_path)
        )

        predicted_label = (
            prediction["label"]
        )

        confidence = (
            prediction["confidence"]
        )

        correct = (
            predicted_label ==
            true_label
        )

        results.append(
            {
                "true": true_label,
                "variant": variant_name,
                "prediction": predicted_label,
                "confidence": confidence,
                "correct": correct
            }
        )

        if index % 20 == 0:

            print(
                f"Processed "
                f"{index}/"
                f"{len(all_variant_files)}"
            )

    except Exception as e:

        print(
            f"ERROR predicting "
            f"{image_path.name}: {e}"
        )


# ============================================================
# OVERALL RESULTS
# ============================================================

print()
print("=" * 70)
print("ROBUSTNESS RESULTS")
print("=" * 70)

variant_names = [
    "original",
    "jpeg_95",
    "jpeg_75",
    "jpeg_50",
    "resize_75",
    "resize_50",
    "blur",
    "noise",
    "crop"
]

summary = []


for variant in variant_names:

    variant_results = [
        r
        for r in results
        if r["variant"] == variant
    ]

    if not variant_results:
        continue

    accuracy = np.mean(
        [
            r["correct"]
            for r in variant_results
        ]
    )

    average_confidence = np.mean(
        [
            r["confidence"]
            for r in variant_results
        ]
    )

    ai_results = [
        r
        for r in variant_results
        if r["true"] == "AI"
    ]

    real_results = [
        r
        for r in variant_results
        if r["true"] == "Real"
    ]

    ai_accuracy = (
        np.mean(
            [
                r["correct"]
                for r in ai_results
            ]
        )
        if ai_results
        else 0
    )

    real_accuracy = (
        np.mean(
            [
                r["correct"]
                for r in real_results
            ]
        )
        if real_results
        else 0
    )

    summary.append(
        (
            variant,
            accuracy,
            ai_accuracy,
            real_accuracy,
            average_confidence
        )
    )


print()

print(
    f"{'Variant':<15}"
    f"{'Accuracy':<12}"
    f"{'AI Acc.':<12}"
    f"{'Real Acc.':<12}"
    f"{'Avg Conf.':<12}"
)

print("-" * 63)


for (
    variant,
    accuracy,
    ai_accuracy,
    real_accuracy,
    average_confidence
) in summary:

    print(
        f"{variant:<15}"
        f"{accuracy * 100:<12.2f}"
        f"{ai_accuracy * 100:<12.2f}"
        f"{real_accuracy * 100:<12.2f}"
        f"{average_confidence:<12.2f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = (
    PROJECT_DIR /
    "evaluation" /
    "robustness_results_v4b.txt"
)

with open(
    results_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "V4-B REAL-WORLD ROBUSTNESS TEST\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"{'Variant':<15}"
        f"{'Accuracy':<12}"
        f"{'AI Acc.':<12}"
        f"{'Real Acc.':<12}"
        f"{'Avg Conf.':<12}\n"
    )

    f.write("-" * 63 + "\n")

    for (
        variant,
        accuracy,
        ai_accuracy,
        real_accuracy,
        average_confidence
    ) in summary:

        f.write(
            f"{variant:<15}"
            f"{accuracy * 100:<12.2f}"
            f"{ai_accuracy * 100:<12.2f}"
            f"{real_accuracy * 100:<12.2f}"
            f"{average_confidence:<12.2f}\n"
        )


print()
print("=" * 70)
print("RESULTS SAVED")
print("=" * 70)

print()
print(results_file)

print()
print("=" * 70)
print("ROBUSTNESS TEST COMPLETE")
print("=" * 70)
# SignalScope Experiment Results

## Overview

This file records the results of the SignalScope model experiments.

The external test set is kept separate from training and is used only to evaluate generalization.

---

# V1 — ResNet18 Baseline

## Training

- Architecture: ResNet18
- Pretrained backbone: Yes
- Backbone training: Frozen except classifier
- Input size: 224 × 224
- Epochs: 10
- Dataset:
  - Real: 1,000
  - AI: 875
- Train/Validation/Test split: 80/10/10

## Internal Test Results

- Accuracy: 92.02%
- Precision: 97.75%
- Recall: 87.00%
- Macro-F1: 92.02%
- ROC-AUC: 0.9744

## Confusion Matrix

```text
[[86,  2],
 [13, 87]]

## V4-B — Robustness Augmentation

Model:
- ResNet18
- Fine-tuned layer3, layer4, and fc
- Same V4-A dataset
- Added JPEG compression augmentation
- Added Gaussian blur augmentation
- Added pixel noise
- Stronger spatial/color augmentation
- 15 epochs
- CUDA / RTX 3050

Best validation accuracy:
- 92.33%

Internal V4-A test:
- Accuracy: 91.92%
- Precision: 90.88%
- Recall: 93.20%
- Macro-F1: 91.92%
- ROC-AUC: 0.9778
- Confusion matrix: [[533, 55], [40, 548]]

Unseen SD3 test:
- Accuracy: 95.93%
- Precision: 92.13%
- Recall: 96.00%
- Macro-F1: 95.47%
- ROC-AUC: 0.9931
- Confusion matrix: [[959, 41], [20, 480]]

External 60-image test:
- Accuracy: 80.00%
- Precision: 95.24%
- Recall: 80.00%
- Macro-F1: 72.05%
- ROC-AUC: 0.8440
- Confusion matrix: [[8, 2], [10, 40]]

Conclusion:
V4-B improved external benchmark performance over V4-A while
retaining strong internal and unseen-generator performance.
V4-B is the current primary model candidate.
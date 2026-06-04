"""
Evaluate DenseNet121 pneumonia model on the CLEANED test set.

Usage:
    cd hms-ai
    python -m ml.evaluation.evaluate_pneumonia
    python -m ml.evaluation.evaluate_pneumonia --checkpoint ml/checkpoints/pneumonia_densenet121_best.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve,
)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from ml.training.config import (
    BATCH_SIZE, BEST_MODEL_PATH, CLASS_NAMES, DATA_DIR, DEVICE,
    IMAGENET_MEAN, IMAGENET_STD, METRICS_DIR, MODEL_VERSION,
    NUM_CLASSES, NUM_WORKERS, TEST_DIR,
)


def load_model(checkpoint_path: Path) -> nn.Module:
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print("Train first: python -m ml.training.train_pneumonia")
        sys.exit(1)

    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)

    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    metrics = ckpt.get("metrics", {})
    version = ckpt.get("model_version", "?")
    print(f"  Loaded: epoch={epoch}, val_f1={metrics.get('f1', '?')}, version={version}")
    return model


@torch.no_grad()
def run_inference(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_labels, all_preds, all_probs_list = [], [], []

    for images, labels in tqdm(loader, desc="  Evaluating"):
        images = images.to(DEVICE)
        if DEVICE.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(images)
        else:
            logits = model(images)

        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        all_labels.extend(labels.numpy())
        all_preds.extend(preds)
        all_probs_list.append(probs)

    return np.array(all_labels), np.array(all_preds), np.vstack(all_probs_list)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pneumonia model on cleaned test set")
    parser.add_argument("--checkpoint", type=str, default=str(BEST_MODEL_PATH))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    print("=" * 64)
    print("  DenseNet121 Pneumonia Evaluation (Cleaned Test Set)")
    print("=" * 64)
    print(f"  Device:     {DEVICE}")
    print(f"  Dataset:    {DATA_DIR}")
    print(f"  Test dir:   {TEST_DIR}")
    print(f"  Checkpoint: {args.checkpoint}")

    # Verify cleaned dataset
    if "chest_xray_cleaned" not in str(DATA_DIR):
        print("ERROR: Must evaluate on cleaned dataset.")
        sys.exit(1)
    if not TEST_DIR.exists():
        print(f"ERROR: Test directory not found: {TEST_DIR}")
        sys.exit(1)

    print()
    model = load_model(Path(args.checkpoint))

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test_ds = datasets.ImageFolder(str(TEST_DIR), transform=test_transforms)
    print(f"  Test set: {len(test_ds)} images")
    print(f"  Classes:  {test_ds.class_to_idx}")

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    # Run
    labels, preds, probs = run_inference(model, test_loader)

    # Metrics
    print("\n--- Classification Report ---\n")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))

    accuracy = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    auc = roc_auc_score(labels, probs[:, 1])

    print("--- Summary ---\n")
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  Precision:   {precision:.4f}")
    print(f"  Recall:      {recall:.4f}  (PNEUMONIA sensitivity)")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  F1-score:    {f1:.4f}")
    print(f"  ROC-AUC:     {auc:.4f}")
    print(f"  Confusion:   TN={tn}  FP={fp}  FN={fn}  TP={tp}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Plots
    if not args.no_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Confusion matrix
            fig, ax = plt.subplots(figsize=(7, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                        annot_kws={"size": 16})
            ax.set_xlabel("Predicted", fontsize=12)
            ax.set_ylabel("Actual", fontsize=12)
            ax.set_title(f"Confusion Matrix -- {MODEL_VERSION}\n"
                         f"Accuracy={accuracy:.3f}  F1={f1:.3f}  Recall={recall:.3f}",
                         fontsize=12, fontweight="bold")
            plt.tight_layout()
            plt.savefig(METRICS_DIR / "confusion_matrix.png", dpi=150)
            plt.close()
            print(f"\n  Saved: confusion_matrix.png")

            # ROC curve
            fpr, tpr, _ = roc_curve(labels, probs[:, 1])
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.plot(fpr, tpr, color="blue", lw=2.5, label=f"AUC = {auc:.4f}")
            ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
            ax.set_xlabel("False Positive Rate", fontsize=12)
            ax.set_ylabel("True Positive Rate", fontsize=12)
            ax.set_title(f"ROC Curve -- {MODEL_VERSION}", fontsize=12, fontweight="bold")
            ax.legend(fontsize=12); ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(METRICS_DIR / "roc_curve.png", dpi=150)
            plt.close()
            print(f"  Saved: roc_curve.png")
        except ImportError:
            print("  matplotlib/seaborn not available -- skipping plots")

    # Evaluation report
    report = f"""# Evaluation Report -- {MODEL_VERSION}

## Model
- Architecture: DenseNet121 (ImageNet pretrained)
- Classifier: Linear(1024, 2)
- Checkpoint: {args.checkpoint}

## Dataset
- Path: {TEST_DIR}
- Test images: {len(test_ds)}
- NORMAL: {tn + fp} images
- PNEUMONIA: {tp + fn} images

## Metrics

| Metric | Value |
|--------|-------|
| Accuracy | {accuracy:.4f} |
| Precision | {precision:.4f} |
| Recall / Sensitivity | {recall:.4f} |
| Specificity | {specificity:.4f} |
| F1-score | {f1:.4f} |
| ROC-AUC | {auc:.4f} |

## Confusion Matrix

|  | Predicted NORMAL | Predicted PNEUMONIA |
|--|-----------------|-------------------|
| Actual NORMAL | {tn} (TN) | {fp} (FP) |
| Actual PNEUMONIA | {fn} (FN) | {tp} (TP) |

## Clinical Assessment

- **Sensitivity (Recall):** {recall:.1%} -- {"ACCEPTABLE" if recall > 0.90 else "NEEDS IMPROVEMENT"} for pneumonia screening
- **Specificity:** {specificity:.1%} -- {"ACCEPTABLE" if specificity > 0.80 else "NEEDS IMPROVEMENT"}
- **False Negatives:** {fn} missed pneumonia cases out of {tp + fn}

## Known Limitations
1. Trained on pediatric chest X-rays only (ages 1-5)
2. Single-center data (Guangzhou)
3. Binary classification only (no bacterial/viral distinction)
4. Compression bias in original dataset partially mitigated by standardization
5. NOT validated for clinical use

## Recommendation
{"Model is acceptable for DEMO/RESEARCH purposes." if recall > 0.85 and auc > 0.85 else "Model needs further training or data improvement."}
This model must NOT be used for actual clinical diagnosis.
"""
    report_path = METRICS_DIR / "evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved: evaluation_report.md")

    print(f"\n{'=' * 64}")
    print(f"  Evaluation complete.")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()

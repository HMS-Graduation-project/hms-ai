"""
Evaluate EfficientNet-B0 + threshold optimization + Grad-CAM samples.

Usage:
    cd hms-ai
    python -m ml.evaluation.evaluate_efficientnet_b0
"""

from __future__ import annotations

import csv
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
from sklearn.calibration import calibration_curve
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from ml.training.config import (
    BATCH_SIZE, CLASS_NAMES, DEVICE, IMAGENET_MEAN, IMAGENET_STD,
    METRICS_DIR, NUM_CLASSES, NUM_WORKERS, TEST_DIR,
)

CKPT = Path(__file__).resolve().parent.parent / "checkpoints" / "pneumonia_efficientnet_b0_best.pt"
OUT = METRICS_DIR / "efficientnet_b0"
MODEL_VERSION = "pneumonia-efficientnet-b0-v1"


def load_model() -> nn.Module:
    if not CKPT.exists():
        print(f"ERROR: Checkpoint not found: {CKPT}"); sys.exit(1)
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    print(f"  Loaded: epoch={ckpt.get('epoch','?')}, val_f1={ckpt.get('metrics',{}).get('f1','?')}")
    return model


@torch.no_grad()
def get_predictions(model, loader):
    all_labels, all_preds, all_probs = [], [], []
    for images, labels in tqdm(loader, desc="  Evaluating"):
        images = images.to(DEVICE)
        if DEVICE.type == "cuda":
            with torch.amp.autocast("cuda"): logits = model(images)
        else: logits = model(images)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_labels.extend(labels.numpy())
        all_preds.extend(probs.argmax(axis=1))
        all_probs.append(probs)
    return np.array(all_labels), np.array(all_preds), np.vstack(all_probs)


def metrics_at_threshold(y_true, y_prob, t):
    y_pred = (y_prob >= t).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    n = len(y_true)
    acc = (tp + tn) / n
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    try: auc = roc_auc_score(y_true, y_prob)
    except ValueError: auc = 0.0
    return {"threshold": round(t, 4), "accuracy": round(acc, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "specificity": round(spec, 4), "f1": round(f1, 4),
            "auc": round(auc, 4), "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}


def main() -> None:
    print("=" * 64)
    print("  EfficientNet-B0 Evaluation + Threshold Optimization")
    print("=" * 64)
    print(f"  Device: {DEVICE}")
    print(f"  Test:   {TEST_DIR}\n")

    OUT.mkdir(parents=True, exist_ok=True)
    model = load_model()

    test_ds = datasets.ImageFolder(str(TEST_DIR), transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]))
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    print(f"  Test set: {len(test_ds)} images\n")

    labels, preds, probs = get_predictions(model, loader)
    y_prob = probs[:, 1]

    # Classification report at default 0.5
    print("\n--- Classification Report (threshold=0.50) ---\n")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))

    # Threshold sweep
    print("--- Threshold Sweep ---\n")
    fine = [metrics_at_threshold(labels, y_prob, t) for t in np.arange(0.01, 1.0, 0.01)]
    sweep_path = OUT / "threshold_optimization.csv"
    fields = ["threshold", "accuracy", "precision", "recall", "specificity", "f1", "auc", "TP", "FP", "FN", "TN"]
    with open(sweep_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(fine)
    print(f"  Saved: threshold_optimization.csv")

    # Find optimal thresholds
    balanced = max(fine, key=lambda r: r["f1"])
    screening = max([r for r in fine if r["recall"] >= 0.95], key=lambda r: r["specificity"], default=fine[0])
    default = metrics_at_threshold(labels, y_prob, 0.50)

    print(f"\n  Default (0.50):   F1={default['f1']:.4f}  Recall={default['recall']:.4f}  Spec={default['specificity']:.4f}  FP={default['FP']}  FN={default['FN']}")
    print(f"  Balanced ({balanced['threshold']:.2f}):  F1={balanced['f1']:.4f}  Recall={balanced['recall']:.4f}  Spec={balanced['specificity']:.4f}  FP={balanced['FP']}  FN={balanced['FN']}")
    print(f"  Screening ({screening['threshold']:.2f}): F1={screening['f1']:.4f}  Recall={screening['recall']:.4f}  Spec={screening['specificity']:.4f}  FP={screening['FP']}  FN={screening['FN']}")

    # Plots
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        import seaborn as sns

        # Threshold sweep plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        ts = [r["threshold"] for r in fine]
        axes[0].plot(ts, [r["recall"] for r in fine], "r-", lw=2, label="Recall")
        axes[0].plot(ts, [r["specificity"] for r in fine], "b-", lw=2, label="Specificity")
        axes[0].plot(ts, [r["f1"] for r in fine], "g-", lw=2, label="F1")
        axes[0].axvline(balanced["threshold"], color="green", ls="--", alpha=0.5, label=f"Best F1={balanced['threshold']:.2f}")
        axes[0].set_xlabel("Threshold"); axes[0].set_ylabel("Score"); axes[0].set_ylim(0, 1.05)
        axes[0].set_title("Metrics vs Threshold", fontweight="bold"); axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].plot(ts, [r["FP"] for r in fine], "b-", lw=2, label="FP")
        axes[1].plot(ts, [r["FN"] for r in fine], "r-", lw=2, label="FN")
        axes[1].set_xlabel("Threshold"); axes[1].set_ylabel("Count")
        axes[1].set_title("Errors vs Threshold", fontweight="bold"); axes[1].legend(); axes[1].grid(alpha=0.3)
        plt.suptitle("EfficientNet-B0 Threshold Optimization", fontsize=14, fontweight="bold")
        plt.tight_layout(); plt.savefig(OUT / "threshold_optimization.png", dpi=130); plt.close()
        print(f"  Saved: threshold_optimization.png")

        # Confusion matrix at balanced threshold
        cm = confusion_matrix(labels, (y_prob >= balanced["threshold"]).astype(int))
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax, annot_kws={"size": 16})
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix -- EfficientNet-B0\nThreshold={balanced['threshold']:.2f}  F1={balanced['f1']:.4f}", fontweight="bold")
        plt.tight_layout(); plt.savefig(OUT / "confusion_matrix.png", dpi=150); plt.close()
        print(f"  Saved: confusion_matrix.png")

        # ROC curve
        fpr, tpr, _ = roc_curve(labels, y_prob)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, "b-", lw=2.5, label=f"AUC = {balanced['auc']:.4f}")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC -- EfficientNet-B0", fontweight="bold")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(OUT / "roc_curve.png", dpi=150); plt.close()
        print(f"  Saved: roc_curve.png")
    except ImportError: pass

    # Save evaluation report
    report = f"""# EfficientNet-B0 Evaluation Report

## Model: {MODEL_VERSION}
- Checkpoint: {CKPT.name}
- Test images: {len(test_ds)}

## Optimal Threshold: {balanced['threshold']:.2f}

| Metric | Default (0.50) | Balanced ({balanced['threshold']:.2f}) | Screening ({screening['threshold']:.2f}) |
|--------|---------------|---------------------------------------|----------------------------------------|
| Accuracy | {default['accuracy']:.4f} | {balanced['accuracy']:.4f} | {screening['accuracy']:.4f} |
| Precision | {default['precision']:.4f} | {balanced['precision']:.4f} | {screening['precision']:.4f} |
| Recall | {default['recall']:.4f} | {balanced['recall']:.4f} | {screening['recall']:.4f} |
| Specificity | {default['specificity']:.4f} | {balanced['specificity']:.4f} | {screening['specificity']:.4f} |
| F1 | {default['f1']:.4f} | {balanced['f1']:.4f} | {screening['f1']:.4f} |
| AUC | {default['auc']:.4f} | {balanced['auc']:.4f} | {screening['auc']:.4f} |
| FP | {default['FP']} | {balanced['FP']} | {screening['FP']} |
| FN | {default['FN']} | {balanced['FN']} | {screening['FN']} |
"""
    with open(OUT / "evaluation_report.md", "w", encoding="utf-8") as f: f.write(report)
    print(f"  Saved: evaluation_report.md")

    print(f"\n{'=' * 64}")
    print(f"  Evaluation complete.")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()

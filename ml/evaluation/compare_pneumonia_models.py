"""
Compare DenseNet121 vs EfficientNet-B0 for pneumonia detection.

Usage:
    cd hms-ai
    python -m ml.evaluation.compare_pneumonia_models
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
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from ml.training.config import (
    BATCH_SIZE, CLASS_NAMES, DEVICE, IMAGENET_MEAN, IMAGENET_STD,
    METRICS_DIR, NUM_CLASSES, NUM_WORKERS, TEST_DIR,
)

CKPT_DENSE = Path(__file__).resolve().parent.parent / "checkpoints" / "pneumonia_densenet121_best.pt"
CKPT_EFF = Path(__file__).resolve().parent.parent / "checkpoints" / "pneumonia_efficientnet_b0_best.pt"
OUT = METRICS_DIR / "model_comparison"


def load_densenet() -> nn.Module:
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    ckpt = torch.load(CKPT_DENSE, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    return model, ckpt


def load_efficientnet() -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    ckpt = torch.load(CKPT_EFF, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    return model, ckpt


@torch.no_grad()
def run_inference(model, loader):
    all_labels, all_probs = [], []
    times = []
    for images, labels in tqdm(loader, desc="  Inference", leave=False):
        images = images.to(DEVICE)
        t0 = time.time()
        if DEVICE.type == "cuda":
            with torch.amp.autocast("cuda"): logits = model(images)
        else: logits = model(images)
        times.append((time.time() - t0) / images.size(0))
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_labels.extend(labels.numpy()); all_probs.extend(probs)
    return np.array(all_labels), np.array(all_probs), np.mean(times) * 1000


def compute_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    n = len(y_true)
    acc = (tp + tn) / n; prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1); f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    try: auc = roc_auc_score(y_true, y_prob)
    except ValueError: auc = 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "specificity": spec,
            "f1": f1, "auc": auc, "FP": fp, "FN": fn, "TP": tp, "TN": tn}


def find_optimal_threshold(y_true, y_prob):
    best_f1 = 0; best_t = 0.5
    for t in np.arange(0.01, 1.0, 0.01):
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        if f1 > best_f1: best_f1 = f1; best_t = round(t, 2)
    return best_t


def main() -> None:
    print("=" * 64)
    print("  Model Comparison: DenseNet121 vs EfficientNet-B0")
    print("=" * 64)

    for ckpt, name in [(CKPT_DENSE, "DenseNet121"), (CKPT_EFF, "EfficientNet-B0")]:
        if not ckpt.exists():
            print(f"ERROR: {name} checkpoint not found: {ckpt}"); sys.exit(1)

    OUT.mkdir(parents=True, exist_ok=True)

    test_ds = datasets.ImageFolder(str(TEST_DIR), transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]))
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    print(f"  Test: {len(test_ds)} images  Device: {DEVICE}\n")

    # DenseNet121
    print("  Loading DenseNet121...")
    dense_model, dense_ckpt = load_densenet()
    dense_params = sum(p.numel() for p in dense_model.parameters())
    dense_size = CKPT_DENSE.stat().st_size / 1024 / 1024
    y_true, dense_prob, dense_ms = run_inference(dense_model, loader)
    dense_thresh = find_optimal_threshold(y_true, dense_prob)
    dense_m = compute_metrics(y_true, dense_prob, dense_thresh)
    del dense_model; torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

    # EfficientNet-B0
    print("  Loading EfficientNet-B0...")
    eff_model, eff_ckpt = load_efficientnet()
    eff_params = sum(p.numel() for p in eff_model.parameters())
    eff_size = CKPT_EFF.stat().st_size / 1024 / 1024
    _, eff_prob, eff_ms = run_inference(eff_model, loader)
    eff_thresh = find_optimal_threshold(y_true, eff_prob)
    eff_m = compute_metrics(y_true, eff_prob, eff_thresh)
    del eff_model; torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

    # Print comparison
    print(f"\n  {'Metric':<20s} {'DenseNet121':>15s} {'EfficientNet-B0':>18s} {'Winner':>10s}")
    print(f"  {'-'*65}")
    comparisons = [
        ("Optimal Threshold", f"{dense_thresh:.2f}", f"{eff_thresh:.2f}", ""),
        ("Accuracy", f"{dense_m['accuracy']:.4f}", f"{eff_m['accuracy']:.4f}",
         "Dense" if dense_m['accuracy'] > eff_m['accuracy'] else "Eff"),
        ("Precision", f"{dense_m['precision']:.4f}", f"{eff_m['precision']:.4f}",
         "Dense" if dense_m['precision'] > eff_m['precision'] else "Eff"),
        ("Recall", f"{dense_m['recall']:.4f}", f"{eff_m['recall']:.4f}",
         "Dense" if dense_m['recall'] > eff_m['recall'] else "Eff"),
        ("Specificity", f"{dense_m['specificity']:.4f}", f"{eff_m['specificity']:.4f}",
         "Dense" if dense_m['specificity'] > eff_m['specificity'] else "Eff"),
        ("F1", f"{dense_m['f1']:.4f}", f"{eff_m['f1']:.4f}",
         "Dense" if dense_m['f1'] > eff_m['f1'] else "Eff"),
        ("AUC-ROC", f"{dense_m['auc']:.4f}", f"{eff_m['auc']:.4f}",
         "Dense" if dense_m['auc'] > eff_m['auc'] else "Eff"),
        ("False Positives", str(dense_m['FP']), str(eff_m['FP']),
         "Dense" if dense_m['FP'] < eff_m['FP'] else "Eff"),
        ("False Negatives", str(dense_m['FN']), str(eff_m['FN']),
         "Dense" if dense_m['FN'] < eff_m['FN'] else "Eff"),
        ("Inference (ms/img)", f"{dense_ms:.1f}", f"{eff_ms:.1f}",
         "Dense" if dense_ms < eff_ms else "Eff"),
        ("Parameters", f"{dense_params/1e6:.1f}M", f"{eff_params/1e6:.1f}M",
         "Eff" if eff_params < dense_params else "Dense"),
        ("Checkpoint Size", f"{dense_size:.1f}MB", f"{eff_size:.1f}MB",
         "Eff" if eff_size < dense_size else "Dense"),
    ]
    for name, d, e, win in comparisons:
        print(f"  {name:<20s} {d:>15s} {e:>18s} {win:>10s}")

    # Save CSV
    csv_rows = [{"metric": n, "densenet121": d, "efficientnet_b0": e, "winner": w} for n, d, e, w in comparisons]
    with open(OUT / "model_comparison_metrics.csv", "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=["metric", "densenet121", "efficientnet_b0", "winner"]).writeheader()
        csv.DictWriter(f, fieldnames=["metric", "densenet121", "efficientnet_b0", "winner"]).writerows(csv_rows)

    # Plots
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # ROC curves
        fpr_d, tpr_d, _ = roc_curve(y_true, dense_prob)
        fpr_e, tpr_e, _ = roc_curve(y_true, eff_prob)
        axes[0].plot(fpr_d, tpr_d, "b-", lw=2, label=f"DenseNet121 (AUC={dense_m['auc']:.3f})")
        axes[0].plot(fpr_e, tpr_e, "r-", lw=2, label=f"EfficientNet-B0 (AUC={eff_m['auc']:.3f})")
        axes[0].plot([0, 1], [0, 1], "--", color="gray")
        axes[0].set_title("ROC Curves", fontweight="bold"); axes[0].legend(); axes[0].grid(alpha=0.3)

        # Metrics comparison bar
        metric_names = ["Accuracy", "Precision", "Recall", "Specificity", "F1"]
        dense_vals = [dense_m["accuracy"], dense_m["precision"], dense_m["recall"], dense_m["specificity"], dense_m["f1"]]
        eff_vals = [eff_m["accuracy"], eff_m["precision"], eff_m["recall"], eff_m["specificity"], eff_m["f1"]]
        x = np.arange(len(metric_names)); w = 0.35
        axes[1].bar(x - w/2, dense_vals, w, label="DenseNet121", color="#4C72B0")
        axes[1].bar(x + w/2, eff_vals, w, label="EfficientNet-B0", color="#DD8452")
        axes[1].set_xticks(x); axes[1].set_xticklabels(metric_names, fontsize=9)
        axes[1].set_ylim(0, 1.05); axes[1].set_title("Metrics Comparison", fontweight="bold")
        axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)

        # Error counts
        err_names = ["False Positives", "False Negatives"]
        axes[2].bar(np.arange(2) - 0.2, [dense_m["FP"], dense_m["FN"]], 0.35, label="DenseNet121", color="#4C72B0")
        axes[2].bar(np.arange(2) + 0.2, [eff_m["FP"], eff_m["FN"]], 0.35, label="EfficientNet-B0", color="#DD8452")
        axes[2].set_xticks(np.arange(2)); axes[2].set_xticklabels(err_names)
        axes[2].set_title("Error Counts", fontweight="bold"); axes[2].legend(); axes[2].grid(axis="y", alpha=0.3)

        plt.suptitle("DenseNet121 vs EfficientNet-B0", fontsize=14, fontweight="bold")
        plt.tight_layout(); plt.savefig(OUT / "model_comparison_charts.png", dpi=130); plt.close()
        print(f"\n  Saved: model_comparison_charts.png")
    except ImportError: pass

    # Clinical report
    better_recall = "DenseNet121" if dense_m["recall"] > eff_m["recall"] else "EfficientNet-B0"
    better_spec = "DenseNet121" if dense_m["specificity"] > eff_m["specificity"] else "EfficientNet-B0"
    better_f1 = "DenseNet121" if dense_m["f1"] > eff_m["f1"] else "EfficientNet-B0"
    fewer_fn = "DenseNet121" if dense_m["FN"] < eff_m["FN"] else "EfficientNet-B0"

    report = f"""# Model Comparison Report

## DenseNet121 vs EfficientNet-B0 for Pneumonia Detection

| Metric | DenseNet121 | EfficientNet-B0 |
|--------|-------------|-----------------|
| Optimal Threshold | {dense_thresh} | {eff_thresh} |
| Accuracy | {dense_m['accuracy']:.4f} | {eff_m['accuracy']:.4f} |
| Precision | {dense_m['precision']:.4f} | {eff_m['precision']:.4f} |
| Recall (Sensitivity) | {dense_m['recall']:.4f} | {eff_m['recall']:.4f} |
| Specificity | {dense_m['specificity']:.4f} | {eff_m['specificity']:.4f} |
| F1-score | {dense_m['f1']:.4f} | {eff_m['f1']:.4f} |
| AUC-ROC | {dense_m['auc']:.4f} | {eff_m['auc']:.4f} |
| False Positives | {dense_m['FP']} | {eff_m['FP']} |
| False Negatives | {dense_m['FN']} | {eff_m['FN']} |
| Inference (ms/img) | {dense_ms:.1f} | {eff_ms:.1f} |
| Parameters | {dense_params/1e6:.1f}M | {eff_params/1e6:.1f}M |
| Checkpoint Size | {dense_size:.1f}MB | {eff_size:.1f}MB |

## Clinical Assessment

1. **Better sensitivity (recall):** {better_recall} ({max(dense_m['recall'], eff_m['recall']):.1%})
2. **Better specificity:** {better_spec} ({max(dense_m['specificity'], eff_m['specificity']):.1%})
3. **Better F1 (overall):** {better_f1} ({max(dense_m['f1'], eff_m['f1']):.4f})
4. **Fewer missed pneumonia (FN):** {fewer_fn} ({min(dense_m['FN'], eff_m['FN'])} missed)
5. **Smaller model:** EfficientNet-B0 ({eff_params/1e6:.1f}M vs {dense_params/1e6:.1f}M)

## Recommendation

**{better_f1}** is recommended as the default model based on overall F1 performance.

For **screening priority** (maximize recall): use {better_recall}.
For **confirmation** (minimize false positives): use {better_spec}.

Both models are for **AI-assisted screening only**. Neither constitutes a final diagnosis.
Clinical decisions must be made by qualified healthcare professionals.

## Known Limitations
- Both trained on pediatric chest X-rays (ages 1-5) from a single center
- Not validated for clinical deployment
- Compression bias partially mitigated by dataset standardization
"""
    with open(OUT / "model_comparison_report.md", "w", encoding="utf-8") as f: f.write(report)
    with open(OUT / "clinical_model_comparison.md", "w", encoding="utf-8") as f: f.write(report)
    print(f"  Saved: model_comparison_report.md")
    print(f"  Saved: clinical_model_comparison.md")

    print(f"\n{'=' * 64}")
    print(f"  Comparison complete.")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()

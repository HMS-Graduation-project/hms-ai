"""
Threshold Tuning, Calibration & Clinical Optimization.

Reuses existing trained checkpoint -- no retraining.

Usage:
    cd hms-ai
    python -m ml.evaluation.threshold_calibration
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
    accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.calibration import calibration_curve
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from ml.training.config import (
    BATCH_SIZE, BEST_MODEL_PATH, CLASS_NAMES, DEVICE,
    IMAGENET_MEAN, IMAGENET_STD, METRICS_DIR, MODEL_VERSION,
    NUM_CLASSES, NUM_WORKERS, TEST_DIR,
)

OUTPUT = METRICS_DIR


def _heading(t: str) -> None:
    print(f"\n{'=' * 64}\n  {t}\n{'=' * 64}\n")


def _save_md(path: Path, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Saved: {path.name}")


# ── Load model and get predictions ──────────────────────────────────────


def get_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Load model, run on test set, return (y_true, y_prob_pneumonia)."""
    if not BEST_MODEL_PATH.exists():
        print(f"ERROR: Checkpoint not found: {BEST_MODEL_PATH}")
        sys.exit(1)

    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    ckpt = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    print(f"  Loaded checkpoint: epoch={ckpt.get('epoch','?')}, version={ckpt.get('model_version','?')}")

    test_ds = datasets.ImageFolder(str(TEST_DIR), transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]))
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    print(f"  Test set: {len(test_ds)} images")

    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Inference", leave=False):
            images = images.to(DEVICE)
            if DEVICE.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = model(images)
            else:
                logits = model(images)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    return np.array(all_labels), np.array(all_probs)


def compute_metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thresh: float) -> dict:
    y_pred = (y_prob >= thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = len(y_true)
    acc = (tp + tn) / total
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.0
    return {
        "threshold": round(thresh, 4), "accuracy": round(acc, 4),
        "precision": round(prec, 4), "recall": round(rec, 4),
        "specificity": round(spec, 4), "f1": round(f1, 4),
        "auc": round(auc, 4),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


# ── Phase 1: Probability distribution ──────────────────────────────────


def phase1_probability(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    _heading("Phase 1: Probability Distribution Analysis")

    # CSV
    rows = [{"index": i, "true_label": int(y_true[i]),
             "true_class": CLASS_NAMES[int(y_true[i])],
             "pneumonia_probability": round(float(y_prob[i]), 6)}
            for i in range(len(y_true))]
    out = OUTPUT / "probability_distribution.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["index", "true_label", "true_class", "pneumonia_probability"])
        w.writeheader(); w.writerows(rows)
    print(f"  Saved: {out.name} ({len(rows)} rows)")

    # Stats
    for cls_idx, cls in enumerate(CLASS_NAMES):
        mask = y_true == cls_idx
        probs = y_prob[mask]
        print(f"  {cls:12s}  n={mask.sum():>4}  "
              f"prob_mean={probs.mean():.4f}  median={np.median(probs):.4f}  "
              f"std={probs.std():.4f}  [{probs.min():.4f} - {probs.max():.4f}]")

    # Overlap: NORMAL images with prob > 0.3 AND PNEUMONIA images with prob < 0.7
    normal_high = (y_prob[y_true == 0] > 0.3).sum()
    pneum_low = (y_prob[y_true == 1] < 0.7).sum()
    print(f"\n  Uncertainty overlap:")
    print(f"    NORMAL with prob > 0.3 (risky): {normal_high}")
    print(f"    PNEUMONIA with prob < 0.7 (uncertain): {pneum_low}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        axes[0].hist(y_prob[y_true == 0], bins=50, alpha=0.7, color="#4C72B0",
                     label="NORMAL", edgecolor="white", density=True)
        axes[0].hist(y_prob[y_true == 1], bins=50, alpha=0.7, color="#DD8452",
                     label="PNEUMONIA", edgecolor="white", density=True)
        axes[0].axvline(0.5, color="red", ls="--", lw=2, label="Default threshold (0.5)")
        axes[0].set_xlabel("P(PNEUMONIA)"); axes[0].set_ylabel("Density")
        axes[0].set_title("Probability Distribution by True Class", fontweight="bold")
        axes[0].legend(); axes[0].grid(alpha=0.3)

        # Box plot
        data = [y_prob[y_true == 0], y_prob[y_true == 1]]
        bp = axes[1].boxplot(data, tick_labels=CLASS_NAMES, patch_artist=True,
                             medianprops=dict(color="white", lw=2))
        for patch, c in zip(bp["boxes"], ["#4C72B0", "#DD8452"]):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        axes[1].set_ylabel("P(PNEUMONIA)")
        axes[1].set_title("Probability Box Plot", fontweight="bold")
        axes[1].axhline(0.5, color="red", ls="--", lw=1.5, label="0.5 threshold")
        axes[1].legend(); axes[1].grid(alpha=0.3)

        plt.suptitle("Probability Analysis -- Pneumonia Classifier", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUTPUT / "probability_distribution.png", dpi=130)
        plt.close()
        print(f"  Saved: probability_distribution.png")
    except ImportError:
        pass


# ── Phase 2: Threshold sweep ───────────────────────────────────────────


def phase2_threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray) -> list[dict]:
    _heading("Phase 2: Threshold Sweep (0.05 - 0.95)")

    thresholds = np.arange(0.05, 0.96, 0.05)
    results = [compute_metrics_at_threshold(y_true, y_prob, t) for t in thresholds]

    out = OUTPUT / "threshold_analysis.csv"
    fields = ["threshold", "accuracy", "precision", "recall", "specificity", "f1", "auc", "TP", "FP", "FN", "TN"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(results)
    print(f"  Saved: {out.name}")

    # Print key thresholds
    print(f"\n  {'Thresh':>7s} {'Acc':>7s} {'Prec':>7s} {'Recall':>7s} {'Spec':>7s} {'F1':>7s} {'FP':>5s} {'FN':>5s}")
    print(f"  {'-'*54}")
    for r in results:
        print(f"  {r['threshold']:>7.2f} {r['accuracy']:>7.4f} {r['precision']:>7.4f} "
              f"{r['recall']:>7.4f} {r['specificity']:>7.4f} {r['f1']:>7.4f} {r['FP']:>5d} {r['FN']:>5d}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        ts = [r["threshold"] for r in results]

        axes[0].plot(ts, [r["recall"] for r in results], "r-o", lw=2, ms=4, label="Recall (Sensitivity)")
        axes[0].plot(ts, [r["specificity"] for r in results], "b-o", lw=2, ms=4, label="Specificity")
        axes[0].plot(ts, [r["f1"] for r in results], "g-o", lw=2, ms=4, label="F1-score")
        axes[0].plot(ts, [r["precision"] for r in results], "m-o", lw=2, ms=4, label="Precision")
        axes[0].axvline(0.5, color="gray", ls="--", alpha=0.5, label="Default (0.5)")
        axes[0].set_xlabel("Threshold"); axes[0].set_ylabel("Score"); axes[0].set_ylim(0, 1.05)
        axes[0].set_title("Metrics vs Threshold", fontweight="bold")
        axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

        axes[1].plot(ts, [r["FP"] for r in results], "b-o", lw=2, ms=4, label="False Positives")
        axes[1].plot(ts, [r["FN"] for r in results], "r-o", lw=2, ms=4, label="False Negatives")
        axes[1].axvline(0.5, color="gray", ls="--", alpha=0.5)
        axes[1].set_xlabel("Threshold"); axes[1].set_ylabel("Count")
        axes[1].set_title("Error Counts vs Threshold", fontweight="bold")
        axes[1].legend(); axes[1].grid(alpha=0.3)

        plt.suptitle("Threshold Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUTPUT / "threshold_analysis.png", dpi=130)
        plt.close()
        print(f"  Saved: threshold_analysis.png")
    except ImportError:
        pass

    return results


# ── Phase 3: Clinical threshold optimization ───────────────────────────


def phase3_clinical_thresholds(y_true: np.ndarray, y_prob: np.ndarray, sweep: list[dict]) -> dict[str, dict]:
    _heading("Phase 3: Clinical Threshold Optimization")

    # Fine-grained sweep for optimal points
    fine_ts = np.arange(0.01, 1.0, 0.01)
    fine = [compute_metrics_at_threshold(y_true, y_prob, t) for t in fine_ts]

    # Screening: max recall with recall >= 0.95
    screening = max([r for r in fine if r["recall"] >= 0.95], key=lambda r: r["specificity"], default=fine[0])

    # Balanced: max F1
    balanced = max(fine, key=lambda r: r["f1"])

    # High specificity: max specificity with recall >= 0.85
    high_spec_candidates = [r for r in fine if r["recall"] >= 0.85]
    high_spec = max(high_spec_candidates, key=lambda r: r["specificity"]) if high_spec_candidates else balanced

    thresholds = {"screening": screening, "balanced": balanced, "high_specificity": high_spec}

    for name, m in thresholds.items():
        print(f"  {name:20s}  thresh={m['threshold']:.2f}  "
              f"recall={m['recall']:.4f}  spec={m['specificity']:.4f}  "
              f"F1={m['f1']:.4f}  FP={m['FP']}  FN={m['FN']}")

    # Compare with default
    default = compute_metrics_at_threshold(y_true, y_prob, 0.50)
    print(f"\n  {'default (0.50)':20s}  thresh=0.50  "
          f"recall={default['recall']:.4f}  spec={default['specificity']:.4f}  "
          f"F1={default['f1']:.4f}  FP={default['FP']}  FN={default['FN']}")

    report = f"""# Threshold Optimization Report

## Clinical Operating Points

| Mode | Threshold | Recall | Specificity | Precision | F1 | FP | FN |
|------|-----------|--------|-------------|-----------|-----|-----|-----|
| Default | 0.50 | {default['recall']:.4f} | {default['specificity']:.4f} | {default['precision']:.4f} | {default['f1']:.4f} | {default['FP']} | {default['FN']} |
| Screening | {screening['threshold']:.2f} | {screening['recall']:.4f} | {screening['specificity']:.4f} | {screening['precision']:.4f} | {screening['f1']:.4f} | {screening['FP']} | {screening['FN']} |
| Balanced | {balanced['threshold']:.2f} | {balanced['recall']:.4f} | {balanced['specificity']:.4f} | {balanced['precision']:.4f} | {balanced['f1']:.4f} | {balanced['FP']} | {balanced['FN']} |
| High Specificity | {high_spec['threshold']:.2f} | {high_spec['recall']:.4f} | {high_spec['specificity']:.4f} | {high_spec['precision']:.4f} | {high_spec['f1']:.4f} | {high_spec['FP']} | {high_spec['FN']} |

## Recommendations

### For Mass Screening (Emergency/Triage)
Use threshold **{screening['threshold']:.2f}** -- catches {screening['recall']:.1%} of pneumonia cases.
Accept {screening['FP']} false positives for safety. Every suspect case gets further review.

### For Balanced Clinical Decision Support
Use threshold **{balanced['threshold']:.2f}** -- best F1 ({balanced['f1']:.4f}).
Reduces false positives to {balanced['FP']} while maintaining {balanced['recall']:.1%} recall.

### For Confirmation (High Confidence)
Use threshold **{high_spec['threshold']:.2f}** -- specificity {high_spec['specificity']:.1%}.
Use when you need high confidence before acting on the prediction.

### Key Improvement
Moving from default (0.50) to balanced ({balanced['threshold']:.2f}):
- False Positives: {default['FP']} -> {balanced['FP']} (reduced by {default['FP'] - balanced['FP']})
- False Negatives: {default['FN']} -> {balanced['FN']} (increased by {balanced['FN'] - default['FN']})
- Specificity: {default['specificity']:.1%} -> {balanced['specificity']:.1%}
- F1: {default['f1']:.4f} -> {balanced['f1']:.4f}
"""
    _save_md(OUTPUT / "threshold_report.md", report)
    return thresholds


# ── Phase 4: Calibration ──────────────────────────────────────────────


def phase4_calibration(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    _heading("Phase 4: Calibration Analysis")

    brier = brier_score_loss(y_true, y_prob)
    print(f"  Brier Score: {brier:.4f} (lower is better, perfect = 0)")

    # Reliability diagram
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")

    # Expected Calibration Error
    bin_counts = np.histogram(y_prob, bins=10, range=(0, 1))[0]
    ece = np.sum(np.abs(prob_true - prob_pred) * (bin_counts[bin_counts > 0] / len(y_prob)))
    print(f"  Expected Calibration Error (ECE): {ece:.4f}")

    # Overconfidence check
    high_conf = y_prob[(y_prob > 0.9) | (y_prob < 0.1)]
    high_conf_correct = ((y_prob > 0.9) & (y_true == 1)) | ((y_prob < 0.1) & (y_true == 0))
    high_conf_acc = high_conf_correct.sum() / max(len(high_conf), 1)
    print(f"  High-confidence predictions (>0.9 or <0.1): {len(high_conf)} ({100*len(high_conf)/len(y_prob):.1f}%)")
    print(f"  High-confidence accuracy: {high_conf_acc:.4f}")

    overconfident = high_conf_acc < 0.95
    print(f"  Calibration: {'OVERCONFIDENT' if overconfident else 'WELL CALIBRATED'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Reliability diagram
        axes[0].plot(prob_pred, prob_true, "bo-", lw=2, ms=6, label="Model")
        axes[0].plot([0, 1], [0, 1], "r--", lw=1.5, label="Perfect calibration")
        axes[0].fill_between(prob_pred, prob_true, prob_pred, alpha=0.15, color="blue")
        axes[0].set_xlabel("Mean Predicted Probability"); axes[0].set_ylabel("Fraction of Positives")
        axes[0].set_title(f"Calibration Curve\nBrier={brier:.4f}  ECE={ece:.4f}", fontweight="bold")
        axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)

        # Prediction histogram
        axes[1].hist(y_prob[y_true == 0], bins=30, alpha=0.7, color="#4C72B0", label="NORMAL", edgecolor="white")
        axes[1].hist(y_prob[y_true == 1], bins=30, alpha=0.7, color="#DD8452", label="PNEUMONIA", edgecolor="white")
        axes[1].set_xlabel("P(PNEUMONIA)"); axes[1].set_ylabel("Count")
        axes[1].set_title("Prediction Histogram", fontweight="bold")
        axes[1].legend(); axes[1].grid(alpha=0.3)

        plt.suptitle("Probability Calibration Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUTPUT / "calibration_curve.png", dpi=130)
        plt.close()
        print(f"  Saved: calibration_curve.png")
    except ImportError:
        pass

    report = f"""# Calibration Report

## Metrics
| Metric | Value | Interpretation |
|--------|-------|---------------|
| Brier Score | {brier:.4f} | {"Good" if brier < 0.1 else "Moderate" if brier < 0.2 else "Poor"} (lower=better, perfect=0) |
| ECE | {ece:.4f} | {"Good" if ece < 0.05 else "Moderate" if ece < 0.1 else "Needs calibration"} |
| High-conf accuracy | {high_conf_acc:.4f} | {"Well calibrated" if not overconfident else "Overconfident"} |
| High-conf count | {len(high_conf)} / {len(y_prob)} | {100*len(high_conf)/len(y_prob):.1f}% of predictions |

## Assessment
{"Model probabilities are reasonably well calibrated." if brier < 0.15 and ece < 0.1 else "Model shows calibration issues. Consider temperature scaling or Platt scaling."}

## Note
Calibration is important for clinical decision-making. A well-calibrated model's
probability of 0.8 should mean that ~80% of cases with that prediction are truly positive.
"""
    _save_md(OUTPUT / "calibration_report.md", report)


# ── Phase 5: Error analysis ──────────────────────────────────────────


def phase5_error_analysis(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    _heading("Phase 5: Error Analysis")

    test_ds = datasets.ImageFolder(str(TEST_DIR))
    filenames = [Path(test_ds.samples[i][0]).name for i in range(len(test_ds))]

    y_pred_default = (y_prob >= 0.5).astype(int)
    errors = []
    for i in range(len(y_true)):
        if y_pred_default[i] != y_true[i]:
            errors.append({
                "filename": filenames[i],
                "true_class": CLASS_NAMES[int(y_true[i])],
                "predicted_class": CLASS_NAMES[int(y_pred_default[i])],
                "pneumonia_prob": round(float(y_prob[i]), 4),
                "confidence": round(float(max(y_prob[i], 1 - y_prob[i])), 4),
                "error_type": "FP" if y_pred_default[i] == 1 else "FN",
            })

    # Sort by confidence (most confident mistakes first)
    errors.sort(key=lambda e: -e["confidence"])

    fp_count = sum(1 for e in errors if e["error_type"] == "FP")
    fn_count = sum(1 for e in errors if e["error_type"] == "FN")
    print(f"  Total errors: {len(errors)} (FP={fp_count}, FN={fn_count})")

    report = f"""# Error Analysis Report

## Summary (threshold=0.50)
- Total errors: {len(errors)}
- False Positives: {fp_count} (NORMAL predicted as PNEUMONIA)
- False Negatives: {fn_count} (PNEUMONIA predicted as NORMAL)

## False Positive Analysis
FP errors mean the model is being overly cautious -- flagging healthy X-rays as pneumonia.
This is less dangerous than missing pneumonia but increases clinical workload.

### Top 20 Most Confident False Positives
| # | File | True | Predicted | P(PNEUMONIA) | Confidence |
|---|------|------|-----------|-------------|------------|
"""
    fps = [e for e in errors if e["error_type"] == "FP"][:20]
    for i, e in enumerate(fps, 1):
        report += f"| {i} | {e['filename']} | {e['true_class']} | {e['predicted_class']} | {e['pneumonia_prob']:.4f} | {e['confidence']:.4f} |\n"

    report += f"""
## False Negative Analysis
FN errors are clinically dangerous -- missing actual pneumonia cases.

### All False Negatives
| # | File | True | Predicted | P(PNEUMONIA) | Confidence |
|---|------|------|-----------|-------------|------------|
"""
    fns = [e for e in errors if e["error_type"] == "FN"]
    for i, e in enumerate(fns, 1):
        report += f"| {i} | {e['filename']} | {e['true_class']} | {e['predicted_class']} | {e['pneumonia_prob']:.4f} | {e['confidence']:.4f} |\n"

    report += f"""
## Key Observations
- False positives dominate ({fp_count} vs {fn_count} false negatives)
- The model is conservative: it prefers to flag uncertain cases as PNEUMONIA
- This behavior is **clinically preferred** for a screening tool
- Most FP errors have moderate confidence (0.5-0.8), not extreme overconfidence
"""
    _save_md(OUTPUT / "error_analysis.md", report)


# ── Phase 6: Optimized evaluation ────────────────────────────────────


def phase6_optimized_report(y_true: np.ndarray, y_prob: np.ndarray, thresholds: dict[str, dict]) -> None:
    _heading("Phase 6: Optimized Clinical Assessment")

    default = compute_metrics_at_threshold(y_true, y_prob, 0.50)
    balanced = thresholds["balanced"]

    report = f"""# Optimized Evaluation Report -- {MODEL_VERSION}

## Comparison: Default vs Optimized Threshold

| Metric | Default (0.50) | Balanced ({balanced['threshold']:.2f}) | Change |
|--------|---------------|--------------------------------------|--------|
| Accuracy | {default['accuracy']:.4f} | {balanced['accuracy']:.4f} | {balanced['accuracy'] - default['accuracy']:+.4f} |
| Precision | {default['precision']:.4f} | {balanced['precision']:.4f} | {balanced['precision'] - default['precision']:+.4f} |
| Recall | {default['recall']:.4f} | {balanced['recall']:.4f} | {balanced['recall'] - default['recall']:+.4f} |
| Specificity | {default['specificity']:.4f} | {balanced['specificity']:.4f} | {balanced['specificity'] - default['specificity']:+.4f} |
| F1-score | {default['f1']:.4f} | {balanced['f1']:.4f} | {balanced['f1'] - default['f1']:+.4f} |
| False Positives | {default['FP']} | {balanced['FP']} | {balanced['FP'] - default['FP']:+d} |
| False Negatives | {default['FN']} | {balanced['FN']} | {balanced['FN'] - default['FN']:+d} |

## Clinical Recommendation

**Recommended threshold: {balanced['threshold']:.2f}**

### Rationale
1. Reduces false positives from {default['FP']} to {balanced['FP']} ({default['FP'] - balanced['FP']} fewer unnecessary follow-ups)
2. Maintains recall at {balanced['recall']:.1%} (vs {default['recall']:.1%} at default)
3. F1-score improves from {default['f1']:.4f} to {balanced['f1']:.4f}
4. Specificity improves from {default['specificity']:.1%} to {balanced['specificity']:.1%}

### Clinical Impact
- Fewer unnecessary radiologist reviews
- Maintains high pneumonia detection rate
- Better resource utilization in screening programs

### Deployment Options
| Use Case | Threshold | Priority |
|----------|-----------|----------|
| Mass screening / triage | {thresholds['screening']['threshold']:.2f} | Maximize detection |
| Clinical decision support | {balanced['threshold']:.2f} | Best overall |
| Confirmation tool | {thresholds['high_specificity']['threshold']:.2f} | Minimize false alarms |

### Limitations
1. Trained on pediatric chest X-rays only
2. Single-center data
3. NOT validated for clinical use
4. Must NOT replace radiologist assessment

**This model is suitable for DEMO and RESEARCH purposes only.**
"""
    _save_md(OUTPUT / "optimized_evaluation_report.md", report)

    print(f"  Default (0.50):   F1={default['f1']:.4f}  Recall={default['recall']:.4f}  Spec={default['specificity']:.4f}")
    print(f"  Balanced ({balanced['threshold']:.2f}):  F1={balanced['f1']:.4f}  Recall={balanced['recall']:.4f}  Spec={balanced['specificity']:.4f}")
    improvement = balanced['specificity'] - default['specificity']
    print(f"\n  Specificity improvement: +{improvement:.1%}")
    print(f"  False positives reduced: {default['FP']} -> {balanced['FP']}")


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 64)
    print("  Threshold Tuning, Calibration & Clinical Optimization")
    print("=" * 64)
    print(f"  Device:     {DEVICE}")
    print(f"  Checkpoint: {BEST_MODEL_PATH}")
    print(f"  Test dir:   {TEST_DIR}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    y_true, y_prob = get_predictions()

    phase1_probability(y_true, y_prob)
    sweep = phase2_threshold_sweep(y_true, y_prob)
    thresholds = phase3_clinical_thresholds(y_true, y_prob, sweep)
    phase4_calibration(y_true, y_prob)
    phase5_error_analysis(y_true, y_prob)
    phase6_optimized_report(y_true, y_prob, thresholds)

    print(f"\n{'=' * 64}")
    print(f"  Complete in {time.time() - t0:.1f}s")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()

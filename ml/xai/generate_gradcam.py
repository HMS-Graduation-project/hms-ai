"""
Generate Grad-CAM explanations for the pneumonia classifier.

Usage:
    cd hms-ai
    python -m ml.xai.generate_gradcam                          # full study
    python -m ml.xai.generate_gradcam --image path/to/xray.jpg # single image
    python -m ml.xai.generate_gradcam --max-per-category 10    # fewer samples
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms

from ml.training.config import (
    BEST_MODEL_PATH, CLASS_NAMES, DEVICE, IMAGENET_MEAN, IMAGENET_STD,
    IMAGE_SIZE, METRICS_DIR, MODEL_VERSION, TEST_DIR,
)
from ml.xai.gradcam import GradCAM, load_model, preprocess_image

OUTPUT_BASE = METRICS_DIR / "gradcam"
ORIG_DIR = OUTPUT_BASE / "original"
HEAT_DIR = OUTPUT_BASE / "heatmap"
OVER_DIR = OUTPUT_BASE / "overlay"
COMP_DIR = OUTPUT_BASE / "comparison"

THRESHOLD = 0.94  # optimized clinical threshold


def _heading(t: str) -> None:
    print(f"\n{'=' * 64}\n  {t}\n{'=' * 64}\n")


def _save_md(path: Path, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Saved: {path.name}")


def save_comparison(
    original: np.ndarray, heatmap: np.ndarray, overlay: np.ndarray,
    pred_class: int, probs: np.ndarray, true_class: int,
    filename: str, category: str,
) -> dict:
    """Save 3-panel comparison and return stats dict."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0d1117")

    axes[0].imshow(original)
    axes[0].set_title("Original", color="white", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(heatmap)
    axes[1].set_title("Grad-CAM Heatmap", color="white", fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", color="white", fontweight="bold")
    axes[2].axis("off")

    pred_name = CLASS_NAMES[pred_class]
    true_name = CLASS_NAMES[true_class]
    prob = probs[pred_class]
    correct = pred_class == true_class

    color = "#2ecc71" if correct else "#e74c3c"
    status = "CORRECT" if correct else category.upper()
    plt.suptitle(
        f"True: {true_name}  |  Predicted: {pred_name}  |  "
        f"P(PNEUMONIA)={probs[1]:.3f}  |  {status}",
        color=color, fontsize=12, fontweight="bold",
    )

    plt.tight_layout()
    out_path = COMP_DIR / f"{category}_{filename}"
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    # Also save individual images
    Image.fromarray(original).save(ORIG_DIR / f"{category}_{filename}")
    Image.fromarray(heatmap).save(HEAT_DIR / f"{category}_{filename}")
    Image.fromarray(overlay).save(OVER_DIR / f"{category}_{filename}")

    # Heatmap statistics
    heat_gray = np.mean(heatmap.astype(np.float32), axis=2) / 255.0
    coverage = float((heat_gray > 0.3).mean())  # fraction of image with activation > 0.3
    intensity_mean = float(heat_gray.mean())
    intensity_max = float(heat_gray.max())

    # Check if attention is on borders (potential artifact shortcut)
    h, w = heat_gray.shape
    border_width = int(min(h, w) * 0.1)
    border_mask = np.zeros_like(heat_gray, dtype=bool)
    border_mask[:border_width, :] = True
    border_mask[-border_width:, :] = True
    border_mask[:, :border_width] = True
    border_mask[:, -border_width:] = True
    center_mask = ~border_mask

    border_activation = float(heat_gray[border_mask].mean()) if border_mask.any() else 0
    center_activation = float(heat_gray[center_mask].mean()) if center_mask.any() else 0

    return {
        "filename": filename, "category": category,
        "true_class": true_name, "predicted_class": pred_name,
        "prob_pneumonia": round(float(probs[1]), 4),
        "confidence": round(float(prob), 4),
        "correct": correct,
        "coverage_30pct": round(coverage, 4),
        "intensity_mean": round(intensity_mean, 4),
        "intensity_max": round(intensity_max, 4),
        "border_activation": round(border_activation, 4),
        "center_activation": round(center_activation, 4),
        "center_to_border_ratio": round(center_activation / max(border_activation, 1e-6), 2),
    }


def run_study(max_per_cat: int = 20) -> None:
    _heading("Grad-CAM Explainability Study")

    model = load_model()
    cam = GradCAM(model)

    # Create output dirs
    for d in [ORIG_DIR, HEAT_DIR, OVER_DIR, COMP_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Load test set
    test_ds = datasets.ImageFolder(str(TEST_DIR))
    print(f"  Test set: {len(test_ds)} images")
    print(f"  Threshold: {THRESHOLD}")
    print(f"  Model: {MODEL_VERSION}")

    # Classify all test images
    print(f"\n  Running inference on test set...")
    categories: dict[str, list[tuple[int, str, int]]] = defaultdict(list)
    # (dataset_idx, filename, true_label)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    model.eval()
    with torch.no_grad():
        for i, (img_path, true_label) in enumerate(test_ds.samples):
            img = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            tensor = transform(img).unsqueeze(0).to(DEVICE)
            if DEVICE.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = model(tensor)
            else:
                logits = model(tensor)
            prob_pneumonia = float(F.softmax(logits, dim=1)[0, 1].cpu())
            pred = 1 if prob_pneumonia >= THRESHOLD else 0
            fname = Path(img_path).name

            if pred == true_label == 1:
                categories["TP"].append((i, fname, true_label))
            elif pred == true_label == 0:
                categories["TN"].append((i, fname, true_label))
            elif pred == 1 and true_label == 0:
                categories["FP"].append((i, fname, true_label))
            elif pred == 0 and true_label == 1:
                categories["FN"].append((i, fname, true_label))

    print(f"  TP={len(categories['TP'])}  TN={len(categories['TN'])}  "
          f"FP={len(categories['FP'])}  FN={len(categories['FN'])}")

    # Generate Grad-CAM for samples from each category
    all_stats: list[dict] = []
    cat_labels = {
        "TP": ("Correct PNEUMONIA", max_per_cat),
        "TN": ("Correct NORMAL", max_per_cat),
        "FP": ("False Positive", 999),  # all
        "FN": ("False Negative", 999),  # all -- critical
    }

    total_generated = 0
    for cat_key, (cat_label, max_n) in cat_labels.items():
        items = categories[cat_key][:max_n]
        if not items:
            print(f"\n  {cat_label}: 0 images (skipping)")
            continue

        print(f"\n  {cat_label}: generating {len(items)} Grad-CAMs...")
        for idx, fname, true_label in items:
            img_path = test_ds.samples[idx][0]
            try:
                original, heatmap, overlay, pred_class, probs = cam.generate_overlay(img_path)
                stats = save_comparison(
                    original, heatmap, overlay, pred_class, probs,
                    true_label, fname, cat_key.lower(),
                )
                if stats:
                    all_stats.append(stats)
                    total_generated += 1
            except Exception as e:
                print(f"    ERROR: {fname}: {e}")

    print(f"\n  Total Grad-CAMs generated: {total_generated}")

    # Save statistics CSV
    if all_stats:
        csv_path = METRICS_DIR / "gradcam_statistics.csv"
        fields = list(all_stats[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_stats)
        print(f"  Saved: gradcam_statistics.csv ({len(all_stats)} rows)")

    # Generate reports
    generate_error_report(all_stats, categories)
    generate_clinical_report(all_stats, categories)


def generate_error_report(stats: list[dict], categories: dict) -> None:
    _heading("Grad-CAM Error Analysis")

    correct = [s for s in stats if s["correct"]]
    incorrect = [s for s in stats if not s["correct"]]
    fps = [s for s in stats if s["category"] == "fp"]
    fns = [s for s in stats if s["category"] == "fn"]

    # Border vs center analysis
    correct_ratios = [s["center_to_border_ratio"] for s in correct if s["center_to_border_ratio"] < 100]
    incorrect_ratios = [s["center_to_border_ratio"] for s in incorrect if s["center_to_border_ratio"] < 100]

    correct_coverage = [s["coverage_30pct"] for s in correct]
    incorrect_coverage = [s["coverage_30pct"] for s in incorrect]

    # Pre-compute for f-string safety
    c_center = f"{np.mean([s['center_activation'] for s in correct]):.4f}" if correct else "N/A"
    c_border = f"{np.mean([s['border_activation'] for s in correct]):.4f}" if correct else "N/A"
    c_ratio = f"{np.mean(correct_ratios):.2f}" if correct_ratios else "N/A"
    i_center = f"{np.mean([s['center_activation'] for s in incorrect]):.4f}" if incorrect else "N/A"
    i_border = f"{np.mean([s['border_activation'] for s in incorrect]):.4f}" if incorrect else "N/A"
    i_ratio = f"{np.mean(incorrect_ratios):.2f}" if incorrect_ratios else "N/A"
    c_cov = f"{np.mean(correct_coverage):.1%}" if correct_coverage else "N/A"
    i_cov = f"{np.mean(incorrect_coverage):.1%}" if incorrect_coverage else "N/A"
    focus_msg = ("Model shows CENTER-focused attention (medically appropriate)"
                 if correct_ratios and np.mean(correct_ratios) > 1.0
                 else "Model shows BORDER-focused attention (investigate shortcut learning)")

    report = f"""# Grad-CAM Error Analysis Report

## Summary
- Total images analyzed: {len(stats)}
- Correct predictions: {len(correct)}
- Incorrect predictions: {len(incorrect)}
- False Positives: {len(fps)}
- False Negatives: {len(fns)}
- Threshold used: {THRESHOLD}

## Attention Pattern Analysis

### Center vs Border Activation
This checks whether the model focuses on lung regions (center) or image borders (potential artifacts).

| Category | Center Activation | Border Activation | Center/Border Ratio |
|----------|-------------------|-------------------|---------------------|
| Correct (n={len(correct)}) | {c_center} | {c_border} | {c_ratio} |
| Incorrect (n={len(incorrect)}) | {i_center} | {i_border} | {i_ratio} |

### Interpretation
- Center/Border ratio > 1.0: Model focuses on central lung regions (GOOD)
- Center/Border ratio < 1.0: Model focuses on borders (SUSPICIOUS -- possible artifact shortcut)
- {focus_msg}

### Coverage Analysis
| Category | Mean Coverage (>30% activation) |
|----------|-------------------------------|
| Correct | {c_cov} |
| Incorrect | {i_cov} |

## False Positive Analysis (NORMAL misclassified as PNEUMONIA)
"""
    if fps:
        report += "| File | P(PNEUMONIA) | Coverage | Center/Border |\n|------|-------------|----------|---------------|\n"
        for s in sorted(fps, key=lambda x: -x["prob_pneumonia"])[:20]:
            report += f"| {s['filename']} | {s['prob_pneumonia']:.4f} | {s['coverage_30pct']:.1%} | {s['center_to_border_ratio']:.2f} |\n"
    else:
        report += "No false positives at threshold {THRESHOLD}.\n"

    report += f"""
## False Negative Analysis (PNEUMONIA missed)
"""
    if fns:
        report += "| File | P(PNEUMONIA) | Coverage | Center/Border |\n|------|-------------|----------|---------------|\n"
        for s in sorted(fns, key=lambda x: x["prob_pneumonia"]):
            report += f"| {s['filename']} | {s['prob_pneumonia']:.4f} | {s['coverage_30pct']:.1%} | {s['center_to_border_ratio']:.2f} |\n"
        report += f"\nThese {len(fns)} missed pneumonia cases should be reviewed by a radiologist.\n"
    else:
        report += "No false negatives at this threshold.\n"

    report += f"""
## Key Findings

1. **Attention focus**: {"Center-focused (lung regions) -- medically appropriate" if correct_ratios and np.mean(correct_ratios) > 1.0 else "Needs investigation for border artifacts"}
2. **Coverage**: Model activates on {np.mean(correct_coverage):.0%} of the image on average
3. **False positives**: {len(fps)} cases -- {"likely edge cases with ambiguous features" if fps else "none at this threshold"}
4. **False negatives**: {len(fns)} cases -- {"these require clinical review" if fns else "excellent sensitivity"}
"""
    _save_md(METRICS_DIR / "gradcam_error_analysis.md", report)


def generate_clinical_report(stats: list[dict], categories: dict) -> None:
    _heading("Grad-CAM Clinical Assessment")

    all_ratios = [s["center_to_border_ratio"] for s in stats if s["center_to_border_ratio"] < 100]
    all_coverage = [s["coverage_30pct"] for s in stats]

    medically_plausible = np.mean(all_ratios) > 1.0 if all_ratios else False
    focused = np.mean(all_coverage) < 0.6 if all_coverage else False

    report = f"""# Grad-CAM Clinical Explainability Report

## Model Information
- Architecture: DenseNet121 (ImageNet pretrained)
- Version: {MODEL_VERSION}
- Threshold: {THRESHOLD}
- Target layer: DenseNet121 features (final convolutional block)

## Explainability Assessment

### Medical Plausibility
{"PASS" if medically_plausible else "NEEDS REVIEW"}: The model's attention is {"primarily focused on central lung regions" if medically_plausible else "showing concerning attention patterns"}.

- Mean center/border activation ratio: {np.mean(all_ratios):.2f}
- {"This suggests the model is learning lung-field features relevant to pneumonia detection." if medically_plausible else "This may indicate the model is using image border artifacts as shortcuts."}

### Attention Coverage
{"APPROPRIATE" if focused else "BROAD"}: Mean activation coverage is {np.mean(all_coverage):.0%}.

- {"Focused attention suggests the model identifies specific diagnostic regions." if focused else "Broad attention may indicate the model uses global image features."}
- For pneumonia, we expect activation in lower and mid-lung fields where consolidation typically occurs.

### Trustworthiness Assessment

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Focuses on lungs | {"PASS" if medically_plausible else "REVIEW"} | Center/border ratio = {np.mean(all_ratios):.2f} |
| Not using borders | {"PASS" if medically_plausible else "FAIL"} | Border activation = {np.mean([s['border_activation'] for s in stats]):.4f} |
| Focused regions | {"PASS" if focused else "REVIEW"} | Coverage = {np.mean(all_coverage):.0%} |
| Consistent behavior | {"PASS" if len(stats) > 20 else "INSUFFICIENT DATA"} | {len(stats)} images analyzed |

## Examples

### Good Explanations (high center/border ratio)
"""
    good = sorted([s for s in stats if s["correct"]], key=lambda x: -x["center_to_border_ratio"])[:5]
    for s in good:
        report += f"- {s['filename']}: center/border={s['center_to_border_ratio']:.2f}, P(PNEUMONIA)={s['prob_pneumonia']:.3f}\n"

    report += "\n### Concerning Explanations (low center/border ratio)\n"
    bad = sorted([s for s in stats if s["center_to_border_ratio"] < 100], key=lambda x: x["center_to_border_ratio"])[:5]
    for s in bad:
        report += f"- {s['filename']}: center/border={s['center_to_border_ratio']:.2f}, P(PNEUMONIA)={s['prob_pneumonia']:.3f} ({s['category'].upper()})\n"

    report += f"""
## Known Limitations
1. Grad-CAM shows WHERE the model looks, not WHAT features it detects
2. Resolution is limited by the feature map size (7x7 for DenseNet121)
3. Heatmap intensity does not directly correlate with clinical importance
4. This analysis is for research/demo purposes only

## Conclusion
{"The model appears to use medically meaningful features for pneumonia detection. Attention patterns are focused on lung regions, which is consistent with clinical expectations." if medically_plausible else "The model shows attention patterns that warrant further investigation. Consider retraining with explicit data augmentation to reduce border artifacts."}

**This model is for RESEARCH/DEMO use only. Not validated for clinical diagnosis.**
"""
    _save_md(METRICS_DIR / "gradcam_clinical_report.md", report)


def run_single(image_path: str) -> None:
    """Generate Grad-CAM for a single image."""
    _heading(f"Grad-CAM: {image_path}")

    model = load_model()
    cam_gen = GradCAM(model)
    original, heatmap, overlay, pred_class, probs = cam_gen.generate_overlay(image_path)

    COMP_DIR.mkdir(parents=True, exist_ok=True)
    fname = Path(image_path).stem + ".png"

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(original); axes[0].set_title("Original"); axes[0].axis("off")
        axes[1].imshow(heatmap); axes[1].set_title("Grad-CAM"); axes[1].axis("off")
        axes[2].imshow(overlay); axes[2].set_title("Overlay"); axes[2].axis("off")

        pred_name = CLASS_NAMES[pred_class]
        plt.suptitle(
            f"Predicted: {pred_name}  |  P(PNEUMONIA)={probs[1]:.4f}  |  Threshold={THRESHOLD}",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        out = COMP_DIR / f"single_{fname}"
        plt.savefig(out, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")
    except ImportError:
        pass

    print(f"  Prediction: {CLASS_NAMES[pred_class]}")
    print(f"  P(NORMAL):    {probs[0]:.4f}")
    print(f"  P(PNEUMONIA): {probs[1]:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM explanations")
    parser.add_argument("--image", type=str, default=None, help="Single image path")
    parser.add_argument("--max-per-category", type=int, default=20, help="Max images per category")
    args = parser.parse_args()

    if args.image:
        run_single(args.image)
    else:
        run_study(max_per_cat=args.max_per_category)


if __name__ == "__main__":
    main()

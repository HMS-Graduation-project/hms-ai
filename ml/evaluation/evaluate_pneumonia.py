"""
Pneumonia Model Evaluation Script.

Evaluates the trained DenseNet121 checkpoint on the test set and
prints classification metrics.

Usage:
    cd hms-ai
    python -m ml.evaluation.evaluate_pneumonia
    python -m ml.evaluation.evaluate_pneumonia --checkpoint ml/checkpoints/pneumonia_densenet121_best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from ml.training.config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    CLASS_NAMES,
    CROP_SIZE,
    DEVICE,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    NUM_WORKERS,
    TEST_DIR,
)


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "eda" / "outputs"


def load_model(checkpoint_path: Path) -> nn.Module:
    """Load the trained model from checkpoint."""
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print("Train the model first: python -m ml.training.train_pneumonia")
        sys.exit(1)

    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    val_acc = checkpoint.get("val_accuracy", "?")
    print(f"  Loaded checkpoint: epoch={epoch}, val_accuracy={val_acc}")

    return model


def build_test_loader() -> DataLoader:
    """Build test set DataLoader."""
    test_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    test_dataset = datasets.ImageFolder(str(TEST_DIR), transform=test_transforms)
    print(f"  Test set: {len(test_dataset)} images")
    print(f"  Class mapping: {test_dataset.class_to_idx}")

    return DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )


@torch.no_grad()
def run_inference(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference on the test set. Returns (all_labels, all_preds, all_probs)."""
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[np.ndarray] = []

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
        all_probs.append(probs)

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.vstack(all_probs),
    )


def plot_confusion_matrix(labels: np.ndarray, preds: np.ndarray) -> None:
    """Plot and save confusion matrix heatmap."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("  [skip] matplotlib/seaborn not installed, cannot plot confusion matrix")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Pneumonia DenseNet121")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_roc_curve(labels: np.ndarray, probs: np.ndarray) -> None:
    """Plot and save ROC curve."""
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
    except ImportError:
        print("  [skip] matplotlib not installed, cannot plot ROC curve")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Use PNEUMONIA class (index 1) probability for ROC
    pneumonia_probs = probs[:, 1]
    fpr, tpr, _ = roc_curve(labels, pneumonia_probs)
    auc = roc_auc_score(labels, pneumonia_probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}", color="blue", linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Pneumonia DenseNet121")
    ax.legend(loc="lower right")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "roc_curve.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate pneumonia DenseNet121 on test set",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=str(BEST_MODEL_PATH),
        help=f"Path to model checkpoint (default: {BEST_MODEL_PATH})",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    print("=" * 60)
    print("  Pneumonia DenseNet121 Evaluation")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Checkpoint: {args.checkpoint}")
    print()

    if not TEST_DIR.exists():
        print(f"ERROR: Test directory not found: {TEST_DIR}")
        sys.exit(1)

    # Load model and data
    model = load_model(Path(args.checkpoint))
    test_loader = build_test_loader()
    print()

    # Run inference
    labels, preds, probs = run_inference(model, test_loader)

    # Metrics
    print("\n--- Classification Report ---\n")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))

    accuracy = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs[:, 1])

    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("--- Summary ---\n")
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  ROC-AUC:     {auc:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  Confusion:   TN={tn}  FP={fp}  FN={fn}  TP={tp}")

    # Plots
    if not args.no_plots:
        print("\n--- Generating Plots ---\n")
        plot_confusion_matrix(labels, preds)
        plot_roc_curve(labels, probs)

    print("\n" + "=" * 60)
    print("  Evaluation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

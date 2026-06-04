"""
Chest X-Ray Dataset Exploratory Data Analysis.

Analyzes the chest_xray dataset structure, class distribution,
image dimensions, and detects corrupted files.

Usage:
    cd hms-ai
    python -m ml.eda.chest_xray_eda
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

from ml.training.config import DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, CLASS_NAMES


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

SPLITS = {
    "train": TRAIN_DIR,
    "val": VAL_DIR,
    "test": TEST_DIR,
}

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def count_images(split_dir: Path) -> dict[str, int]:
    """Count images per class in a split directory."""
    counts: dict[str, int] = {}
    for class_name in CLASS_NAMES:
        class_dir = split_dir / class_name
        if class_dir.exists():
            counts[class_name] = sum(
                1 for f in class_dir.iterdir()
                if f.suffix.lower() in IMAGE_EXTENSIONS
            )
        else:
            counts[class_name] = 0
    return counts


def detect_corrupted(split_dir: Path) -> list[str]:
    """Find image files that cannot be opened by PIL."""
    corrupted: list[str] = []
    for class_name in CLASS_NAMES:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            continue
        for f in class_dir.iterdir():
            if f.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                img = Image.open(f)
                img.verify()
            except Exception:
                corrupted.append(str(f))
    return corrupted


def sample_dimensions(split_dir: Path, n: int = 50) -> list[tuple[int, int]]:
    """Read dimensions of up to n images from a split."""
    dims: list[tuple[int, int]] = []
    for class_name in CLASS_NAMES:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            continue
        count = 0
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                with Image.open(f) as img:
                    dims.append(img.size)
            except Exception:
                pass
            count += 1
            if count >= n:
                break
    return dims


def plot_class_distribution(all_counts: dict[str, dict[str, int]]) -> None:
    """Bar chart of class distribution by split. Saves to outputs/."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip] matplotlib not installed, cannot generate plots")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = list(all_counts.keys())
    x_positions = range(len(splits))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    normal_counts = [all_counts[s].get("NORMAL", 0) for s in splits]
    pneumonia_counts = [all_counts[s].get("PNEUMONIA", 0) for s in splits]

    bars1 = ax.bar(
        [x - width / 2 for x in x_positions], normal_counts, width, label="NORMAL", color="#4CAF50"
    )
    bars2 = ax.bar(
        [x + width / 2 for x in x_positions], pneumonia_counts, width, label="PNEUMONIA", color="#F44336"
    )

    ax.set_xlabel("Split")
    ax.set_ylabel("Number of Images")
    ax.set_title("Chest X-Ray Dataset: Class Distribution by Split")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(splits)
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "class_distribution.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_sample_images() -> None:
    """Grid of sample images from train set. Saves to outputs/."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle("Sample Chest X-Ray Images", fontsize=14)

    for row, class_name in enumerate(CLASS_NAMES):
        class_dir = TRAIN_DIR / class_name
        if not class_dir.exists():
            continue
        files = sorted(f for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)[:4]
        for col, f in enumerate(files):
            try:
                img = Image.open(f).convert("L")
                axes[row, col].imshow(img, cmap="gray")
                axes[row, col].set_title(f"{class_name}", fontsize=10)
            except Exception:
                axes[row, col].set_title("Error")
            axes[row, col].axis("off")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "sample_images.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chest X-Ray Dataset EDA")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    args = parser.parse_args()

    print("=" * 60)
    print("  Chest X-Ray Dataset — Exploratory Data Analysis")
    print("=" * 60)
    print(f"\n  Dataset root: {DATA_DIR}")

    if not DATA_DIR.exists():
        print(f"\n  ERROR: Dataset directory not found: {DATA_DIR}")
        print("  Download the chest_xray dataset and place it at the above path.")
        sys.exit(1)

    # 1. Count images per class per split
    print("\n--- Image Counts ---\n")
    all_counts: dict[str, dict[str, int]] = {}
    grand_total = 0

    for split_name, split_dir in SPLITS.items():
        counts = count_images(split_dir)
        all_counts[split_name] = counts
        total = sum(counts.values())
        grand_total += total
        ratio = counts["PNEUMONIA"] / counts["NORMAL"] if counts["NORMAL"] > 0 else float("inf")
        print(f"  {split_name:6s}  NORMAL={counts['NORMAL']:>5d}  PNEUMONIA={counts['PNEUMONIA']:>5d}  Total={total:>5d}  Ratio={ratio:.2f}:1")

    print(f"\n  Grand total: {grand_total} images")

    # 2. Detect corrupted files
    print("\n--- Corrupted File Check ---\n")
    total_corrupted = 0
    for split_name, split_dir in SPLITS.items():
        corrupted = detect_corrupted(split_dir)
        total_corrupted += len(corrupted)
        if corrupted:
            print(f"  {split_name}: {len(corrupted)} corrupted file(s)")
            for f in corrupted[:5]:
                print(f"    - {f}")
        else:
            print(f"  {split_name}: all files OK")

    if total_corrupted == 0:
        print("\n  No corrupted files detected.")

    # 3. Image dimensions
    print("\n--- Image Dimensions (sampled) ---\n")
    for split_name, split_dir in SPLITS.items():
        dims = sample_dimensions(split_dir, n=50)
        if dims:
            widths = [d[0] for d in dims]
            heights = [d[1] for d in dims]
            print(
                f"  {split_name:6s}  "
                f"Width: {min(widths):>4d}–{max(widths):<4d} (avg {sum(widths)//len(widths)})  "
                f"Height: {min(heights):>4d}–{max(heights):<4d} (avg {sum(heights)//len(heights)})"
            )

    # 4. Validation set warning
    val_total = sum(all_counts.get("val", {}).values())
    if val_total < 50:
        print(f"\n  WARNING: Validation set has only {val_total} images.")
        print("  Validation metrics will be noisy. Use test set for reliable evaluation.")

    # 5. Plots
    if not args.no_plots:
        print("\n--- Generating Plots ---\n")
        plot_class_distribution(all_counts)
        plot_sample_images()

    print("\n" + "=" * 60)
    print("  EDA complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

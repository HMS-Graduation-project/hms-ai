"""
Chest X-Ray Dataset — Professional Medical Imaging EDA.

Comprehensive exploratory data analysis for the chest_xray dataset
covering inventory, quality, statistics, duplicates, leakage, and
visual inspection.

Usage:
    cd hms-ai
    python -m ml.eda.chest_xray_eda
    python -m ml.eda.chest_xray_eda --no-plots
    python -m ml.eda.chest_xray_eda --no-hash        # skip slow duplicate hash scan
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageStat

from ml.training.config import DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, CLASS_NAMES

# ── Constants ─────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

SPLITS: dict[str, Path] = {
    "train": TRAIN_DIR,
    "val": VAL_DIR,
    "test": TEST_DIR,
}

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# ── Helpers ───────────────────────────────────────────────────────────────


def _iter_images(split_dir: Path):
    """Yield (class_name, file_path) for every image in a split."""
    for class_name in CLASS_NAMES:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            continue
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() in IMAGE_EXTENSIONS:
                yield class_name, f


def _safe_open(path: Path) -> Image.Image | None:
    """Open an image, return None if corrupted."""
    try:
        img = Image.open(path)
        img.load()  # force full decode
        return img
    except Exception:
        return None


def _file_hash(path: Path) -> str:
    """SHA-256 of file contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}\n")


# ── Image-level statistics (computed once per image) ──────────────────────


class ImageRecord:
    """Stats for a single image file."""

    __slots__ = (
        "split", "class_name", "path", "width", "height", "aspect_ratio",
        "file_size_kb", "mode", "brightness", "contrast", "sharpness",
        "corrupted", "file_hash",
    )

    def __init__(self, split: str, class_name: str, path: Path):
        self.split = split
        self.class_name = class_name
        self.path = path
        self.width: int = 0
        self.height: int = 0
        self.aspect_ratio: float = 0.0
        self.file_size_kb: float = path.stat().st_size / 1024
        self.mode: str = ""
        self.brightness: float = 0.0
        self.contrast: float = 0.0
        self.sharpness: float = 0.0
        self.corrupted: bool = False
        self.file_hash: str = ""

    def analyse(self, compute_hash: bool = True) -> None:
        img = _safe_open(self.path)
        if img is None:
            self.corrupted = True
            return

        self.width, self.height = img.size
        self.aspect_ratio = round(self.width / max(self.height, 1), 3)
        self.mode = img.mode

        # Convert to grayscale for intensity analysis
        gray = img.convert("L")
        stat = ImageStat.Stat(gray)
        self.brightness = round(stat.mean[0], 2)
        self.contrast = round(stat.stddev[0], 2)

        # Sharpness via Laplacian variance
        arr = np.array(gray, dtype=np.float64)
        laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        from scipy.signal import convolve2d  # type: ignore[import-untyped]
        try:
            lap = convolve2d(arr, laplacian, mode="valid")
            self.sharpness = round(float(np.var(lap)), 2)
        except Exception:
            # scipy may not be installed; fallback via PIL edge filter
            edge = gray.filter(ImageFilter.FIND_EDGES)
            self.sharpness = round(float(ImageStat.Stat(edge).stddev[0]), 2)

        img.close()

        if compute_hash:
            self.file_hash = _file_hash(self.path)


# ══════════════════════════════════════════════════════════════════════════
#  Analysis Sections
# ══════════════════════════════════════════════════════════════════════════


def run_inventory(records: list[ImageRecord]) -> dict[str, dict[str, int]]:
    """Section 1: Dataset inventory — counts per split/class."""
    _section("1. Dataset Inventory")

    counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        counts[split] = {"NORMAL": 0, "PNEUMONIA": 0}
    for r in records:
        counts[r.split][r.class_name] += 1

    grand = 0
    for split in SPLITS:
        c = counts[split]
        total = sum(c.values())
        grand += total
        ratio = c["PNEUMONIA"] / max(c["NORMAL"], 1)
        pct_pneumonia = c["PNEUMONIA"] / max(total, 1) * 100
        print(
            f"  {split:6s}  NORMAL={c['NORMAL']:>5d}  PNEUMONIA={c['PNEUMONIA']:>5d}  "
            f"Total={total:>5d}  Ratio={ratio:.2f}:1  Pneumonia%={pct_pneumonia:.1f}%"
        )

    print(f"\n  Grand total: {grand:,} images")
    return counts


def run_corrupted(records: list[ImageRecord]) -> list[ImageRecord]:
    """Section 2: Corrupted image detection."""
    _section("2. Corrupted Image Detection")

    bad = [r for r in records if r.corrupted]
    if not bad:
        print("  No corrupted files detected across all splits.")
    else:
        print(f"  Found {len(bad)} corrupted file(s):")
        for r in bad:
            print(f"    [{r.split}/{r.class_name}] {r.path.name}")
    return bad


def run_dimensions(records: list[ImageRecord]) -> None:
    """Section 3: Image dimension analysis."""
    _section("3. Image Dimensions")

    valid = [r for r in records if not r.corrupted]
    for split in SPLITS:
        subset = [r for r in valid if r.split == split]
        if not subset:
            continue
        ws = [r.width for r in subset]
        hs = [r.height for r in subset]
        print(
            f"  {split:6s}  Width: {min(ws):>5d} – {max(ws):<5d} (avg {int(np.mean(ws)):>5d})  "
            f"Height: {min(hs):>5d} – {max(hs):<5d} (avg {int(np.mean(hs)):>5d})  "
            f"n={len(subset)}"
        )

    # unique sizes
    unique_sizes = set((r.width, r.height) for r in valid)
    print(f"\n  Unique (W x H) combinations: {len(unique_sizes)}")
    if len(unique_sizes) <= 10:
        for w, h in sorted(unique_sizes):
            cnt = sum(1 for r in valid if r.width == w and r.height == h)
            print(f"    {w}x{h}: {cnt} images")


def run_aspect_ratios(records: list[ImageRecord]) -> None:
    """Section 4: Aspect ratio analysis."""
    _section("4. Aspect Ratio Analysis")

    valid = [r for r in records if not r.corrupted]
    ratios = [r.aspect_ratio for r in valid]
    square = sum(1 for r in ratios if 0.95 <= r <= 1.05)
    landscape = sum(1 for r in ratios if r > 1.05)
    portrait = sum(1 for r in ratios if r < 0.95)

    print(f"  Min ratio:   {min(ratios):.3f}")
    print(f"  Max ratio:   {max(ratios):.3f}")
    print(f"  Mean ratio:  {np.mean(ratios):.3f}")
    print(f"  Square (~1.0):   {square:>5d}  ({square / len(ratios) * 100:.1f}%)")
    print(f"  Landscape (>1):  {landscape:>5d}  ({landscape / len(ratios) * 100:.1f}%)")
    print(f"  Portrait (<1):   {portrait:>5d}  ({portrait / len(ratios) * 100:.1f}%)")


def run_file_sizes(records: list[ImageRecord]) -> None:
    """Section 5: File size analysis."""
    _section("5. File Size Analysis")

    valid = [r for r in records if not r.corrupted]
    for split in SPLITS:
        subset = [r for r in valid if r.split == split]
        if not subset:
            continue
        sizes = [r.file_size_kb for r in subset]
        total_mb = sum(sizes) / 1024
        print(
            f"  {split:6s}  Min={min(sizes):>7.1f} KB  Max={max(sizes):>7.1f} KB  "
            f"Avg={np.mean(sizes):>7.1f} KB  Total={total_mb:>7.1f} MB"
        )

    all_sizes = [r.file_size_kb for r in valid]
    total_gb = sum(all_sizes) / 1024 / 1024
    print(f"\n  Dataset total: {total_gb:.2f} GB")

    # Outlier detection (< 1 KB or > 1 MB)
    tiny = [r for r in valid if r.file_size_kb < 1.0]
    huge = [r for r in valid if r.file_size_kb > 1024.0]
    if tiny:
        print(f"\n  WARNING: {len(tiny)} files < 1 KB (possibly empty/corrupt):")
        for r in tiny[:5]:
            print(f"    [{r.split}/{r.class_name}] {r.path.name} ({r.file_size_kb:.1f} KB)")
    if huge:
        print(f"\n  NOTE: {len(huge)} files > 1 MB")


def run_brightness_contrast(records: list[ImageRecord]) -> None:
    """Section 6: Brightness and contrast analysis."""
    _section("6. Brightness & Contrast Analysis")

    valid = [r for r in records if not r.corrupted]

    for class_name in CLASS_NAMES:
        subset = [r for r in valid if r.class_name == class_name]
        if not subset:
            continue
        brights = [r.brightness for r in subset]
        contrasts = [r.contrast for r in subset]
        print(f"  {class_name:10s}  Brightness: mean={np.mean(brights):>6.1f}  std={np.std(brights):>5.1f}  "
              f"Contrast: mean={np.mean(contrasts):>6.1f}  std={np.std(contrasts):>5.1f}")

    # Per-split
    print()
    for split in SPLITS:
        subset = [r for r in valid if r.split == split]
        if not subset:
            continue
        brights = [r.brightness for r in subset]
        print(f"  {split:6s}  Brightness: mean={np.mean(brights):>6.1f}  range=[{min(brights):.0f} – {max(brights):.0f}]")

    # Dark/bright outliers
    very_dark = [r for r in valid if r.brightness < 30]
    very_bright = [r for r in valid if r.brightness > 220]
    if very_dark:
        print(f"\n  WARNING: {len(very_dark)} very dark images (brightness < 30)")
    if very_bright:
        print(f"  WARNING: {len(very_bright)} very bright images (brightness > 220)")


def run_sharpness(records: list[ImageRecord]) -> None:
    """Section 7: Blur / sharpness estimation."""
    _section("7. Sharpness / Blur Estimation")

    valid = [r for r in records if not r.corrupted]
    for class_name in CLASS_NAMES:
        subset = [r for r in valid if r.class_name == class_name]
        if not subset:
            continue
        sharps = [r.sharpness for r in subset]
        print(f"  {class_name:10s}  Sharpness: mean={np.mean(sharps):>10.1f}  "
              f"min={min(sharps):>8.1f}  max={max(sharps):>10.1f}")

    all_sharps = [r.sharpness for r in valid]
    p5 = np.percentile(all_sharps, 5)
    blurry = [r for r in valid if r.sharpness < p5]
    print(f"\n  5th percentile sharpness: {p5:.1f}")
    print(f"  Potentially blurry images (below 5th pctile): {len(blurry)}")
    if blurry:
        for r in blurry[:5]:
            print(f"    [{r.split}/{r.class_name}] {r.path.name}  sharpness={r.sharpness:.1f}")


def run_duplicate_filenames(records: list[ImageRecord]) -> None:
    """Section 8: Duplicate filename detection."""
    _section("8. Duplicate Filename Detection")

    name_map: dict[str, list[ImageRecord]] = defaultdict(list)
    for r in records:
        name_map[r.path.name].append(r)

    dupes = {name: recs for name, recs in name_map.items() if len(recs) > 1}
    if not dupes:
        print("  No duplicate filenames across splits.")
    else:
        print(f"  {len(dupes)} filename(s) appear in multiple locations:")
        for name, recs in sorted(dupes.items())[:20]:
            locs = ", ".join(f"{r.split}/{r.class_name}" for r in recs)
            print(f"    {name} → {locs}")
        if len(dupes) > 20:
            print(f"    ... and {len(dupes) - 20} more")


def run_exact_duplicates(records: list[ImageRecord]) -> dict[str, list[ImageRecord]]:
    """Section 9: Exact duplicate detection via SHA-256."""
    _section("9. Exact Duplicate Detection (SHA-256)")

    hashed = [r for r in records if r.file_hash and not r.corrupted]
    if not hashed:
        print("  Skipped (no hashes computed — run without --no-hash)")
        return {}

    hash_map: dict[str, list[ImageRecord]] = defaultdict(list)
    for r in hashed:
        hash_map[r.file_hash].append(r)

    dupes = {h: recs for h, recs in hash_map.items() if len(recs) > 1}
    total_dup_images = sum(len(recs) - 1 for recs in dupes.values())

    if not dupes:
        print("  No exact duplicate images found.")
    else:
        print(f"  {len(dupes)} duplicate group(s), {total_dup_images} redundant image(s):")
        for h, recs in sorted(dupes.items(), key=lambda x: -len(x[1]))[:15]:
            paths = [f"{r.split}/{r.class_name}/{r.path.name}" for r in recs]
            print(f"    [{len(recs)}x] {paths[0]}")
            for p in paths[1:]:
                print(f"         = {p}")
        if len(dupes) > 15:
            print(f"    ... and {len(dupes) - 15} more groups")
    return dupes


def run_leakage_detection(records: list[ImageRecord]) -> None:
    """Section 10: Data leakage detection across splits."""
    _section("10. Data Leakage Detection (Train ↔ Val ↔ Test)")

    hashed = [r for r in records if r.file_hash and not r.corrupted]
    if not hashed:
        print("  Skipped (no hashes computed)")
        return

    split_hashes: dict[str, set[str]] = defaultdict(set)
    hash_to_rec: dict[str, ImageRecord] = {}
    for r in hashed:
        split_hashes[r.split].add(r.file_hash)
        hash_to_rec[f"{r.split}:{r.file_hash}"] = r

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    found_leakage = False

    for s1, s2 in pairs:
        overlap = split_hashes.get(s1, set()) & split_hashes.get(s2, set())
        if overlap:
            found_leakage = True
            print(f"  LEAKAGE: {len(overlap)} identical image(s) in both {s1} and {s2}:")
            for h in sorted(overlap)[:10]:
                r1 = hash_to_rec.get(f"{s1}:{h}")
                r2 = hash_to_rec.get(f"{s2}:{h}")
                n1 = f"{r1.class_name}/{r1.path.name}" if r1 else "?"
                n2 = f"{r2.class_name}/{r2.path.name}" if r2 else "?"
                print(f"    {s1}/{n1}  ==  {s2}/{n2}")
        else:
            print(f"  {s1} ↔ {s2}: no leakage detected")

    if not found_leakage:
        print("\n  No cross-split data leakage detected.")


def run_class_comparison(records: list[ImageRecord]) -> None:
    """Section 11: NORMAL vs PNEUMONIA statistical comparison."""
    _section("11. Class Comparison: NORMAL vs PNEUMONIA")

    valid = [r for r in records if not r.corrupted]

    for class_name in CLASS_NAMES:
        subset = [r for r in valid if r.class_name == class_name]
        if not subset:
            continue
        brights = [r.brightness for r in subset]
        contrasts = [r.contrast for r in subset]
        sharps = [r.sharpness for r in subset]
        sizes = [r.file_size_kb for r in subset]

        print(f"  {class_name} (n={len(subset)})")
        print(f"    Brightness  mean={np.mean(brights):>6.1f}  median={np.median(brights):>6.1f}  std={np.std(brights):>5.1f}")
        print(f"    Contrast    mean={np.mean(contrasts):>6.1f}  median={np.median(contrasts):>6.1f}  std={np.std(contrasts):>5.1f}")
        print(f"    Sharpness   mean={np.mean(sharps):>10.1f}  median={np.median(sharps):>10.1f}")
        print(f"    File size   mean={np.mean(sizes):>7.1f} KB  median={np.median(sizes):>7.1f} KB")
        print()


# ── Plots ─────────────────────────────────────────────────────────────────


def _get_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def plot_class_distribution(counts: dict[str, dict[str, int]]) -> None:
    plt = _get_plt()
    if not plt:
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = list(counts.keys())
    x = np.arange(len(splits))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    normal = [counts[s]["NORMAL"] for s in splits]
    pneumonia = [counts[s]["PNEUMONIA"] for s in splits]

    b1 = ax.bar(x - w / 2, normal, w, label="NORMAL", color="#4CAF50")
    b2 = ax.bar(x + w / 2, pneumonia, w, label="PNEUMONIA", color="#F44336")
    ax.bar_label(b1, padding=3, fontsize=9)
    ax.bar_label(b2, padding=3, fontsize=9)

    ax.set_xlabel("Split")
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution by Split")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=150)
    plt.close()
    print(f"  Saved: class_distribution.png")


def plot_dimension_scatter(records: list[ImageRecord]) -> None:
    plt = _get_plt()
    if not plt:
        return

    valid = [r for r in records if not r.corrupted]
    fig, ax = plt.subplots(figsize=(8, 6))
    for class_name, color in [("NORMAL", "#4CAF50"), ("PNEUMONIA", "#F44336")]:
        subset = [r for r in valid if r.class_name == class_name]
        ax.scatter(
            [r.width for r in subset], [r.height for r in subset],
            alpha=0.3, s=8, label=class_name, color=color,
        )
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title("Image Dimensions")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dimension_scatter.png", dpi=150)
    plt.close()
    print(f"  Saved: dimension_scatter.png")


def plot_brightness_histograms(records: list[ImageRecord]) -> None:
    plt = _get_plt()
    if not plt:
        return

    valid = [r for r in records if not r.corrupted]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, class_name in enumerate(CLASS_NAMES):
        subset = [r for r in valid if r.class_name == class_name]
        brights = [r.brightness for r in subset]
        color = "#4CAF50" if class_name == "NORMAL" else "#F44336"
        axes[idx].hist(brights, bins=40, color=color, alpha=0.8, edgecolor="white")
        axes[idx].set_title(f"{class_name} — Brightness Distribution")
        axes[idx].set_xlabel("Mean Pixel Intensity")
        axes[idx].set_ylabel("Count")
        axes[idx].axvline(np.mean(brights), color="black", linestyle="--", label=f"Mean={np.mean(brights):.0f}")
        axes[idx].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "brightness_histograms.png", dpi=150)
    plt.close()
    print(f"  Saved: brightness_histograms.png")


def plot_contrast_comparison(records: list[ImageRecord]) -> None:
    plt = _get_plt()
    if not plt:
        return

    valid = [r for r in records if not r.corrupted]
    fig, ax = plt.subplots(figsize=(8, 5))

    data = []
    labels = []
    colors = ["#4CAF50", "#F44336"]
    for class_name in CLASS_NAMES:
        subset = [r for r in valid if r.class_name == class_name]
        data.append([r.contrast for r in subset])
        labels.append(class_name)

    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Contrast (Std Dev of Pixel Intensity)")
    ax.set_title("Contrast Distribution: NORMAL vs PNEUMONIA")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "contrast_boxplot.png", dpi=150)
    plt.close()
    print(f"  Saved: contrast_boxplot.png")


def plot_file_size_distribution(records: list[ImageRecord]) -> None:
    plt = _get_plt()
    if not plt:
        return

    valid = [r for r in records if not r.corrupted]
    fig, ax = plt.subplots(figsize=(8, 5))
    sizes = [r.file_size_kb for r in valid]
    ax.hist(sizes, bins=50, color="#2196F3", alpha=0.8, edgecolor="white")
    ax.set_xlabel("File Size (KB)")
    ax.set_ylabel("Count")
    ax.set_title("File Size Distribution")
    ax.axvline(np.mean(sizes), color="red", linestyle="--", label=f"Mean={np.mean(sizes):.0f} KB")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "file_size_distribution.png", dpi=150)
    plt.close()
    print(f"  Saved: file_size_distribution.png")


def plot_sample_grid(records: list[ImageRecord]) -> None:
    plt = _get_plt()
    if not plt:
        return

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Sample Chest X-Ray Images", fontsize=14)

    train_records = [r for r in records if r.split == "train" and not r.corrupted]
    for row, class_name in enumerate(CLASS_NAMES):
        subset = [r for r in train_records if r.class_name == class_name][:5]
        for col, r in enumerate(subset):
            img = _safe_open(r.path)
            if img:
                axes[row, col].imshow(img.convert("L"), cmap="gray")
                axes[row, col].set_title(f"{class_name}\n{r.width}x{r.height}", fontsize=8)
                img.close()
            axes[row, col].axis("off")
        # blank unused cols
        for col in range(len(subset), 5):
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sample_grid.png", dpi=150)
    plt.close()
    print(f"  Saved: sample_grid.png")


def plot_sharpness_distribution(records: list[ImageRecord]) -> None:
    plt = _get_plt()
    if not plt:
        return

    valid = [r for r in records if not r.corrupted]
    fig, ax = plt.subplots(figsize=(8, 5))

    for class_name, color in [("NORMAL", "#4CAF50"), ("PNEUMONIA", "#F44336")]:
        subset = [r for r in valid if r.class_name == class_name]
        sharps = [r.sharpness for r in subset]
        # clip extreme outliers for readable histogram
        p99 = np.percentile(sharps, 99)
        clipped = [min(s, p99) for s in sharps]
        ax.hist(clipped, bins=40, alpha=0.5, label=class_name, color=color, edgecolor="white")

    ax.set_xlabel("Sharpness (Laplacian Variance)")
    ax.set_ylabel("Count")
    ax.set_title("Sharpness Distribution by Class")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sharpness_distribution.png", dpi=150)
    plt.close()
    print(f"  Saved: sharpness_distribution.png")


# ── CSV Export ────────────────────────────────────────────────────────────


def export_csv(records: list[ImageRecord]) -> None:
    """Export per-image statistics to CSV."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "image_stats.csv"

    fields = [
        "split", "class", "filename", "width", "height", "aspect_ratio",
        "file_size_kb", "mode", "brightness", "contrast", "sharpness",
        "corrupted",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "split": r.split,
                "class": r.class_name,
                "filename": r.path.name,
                "width": r.width,
                "height": r.height,
                "aspect_ratio": r.aspect_ratio,
                "file_size_kb": round(r.file_size_kb, 2),
                "mode": r.mode,
                "brightness": r.brightness,
                "contrast": r.contrast,
                "sharpness": r.sharpness,
                "corrupted": r.corrupted,
            })

    print(f"  Saved: {out_path} ({len(records)} rows)")


# ── Final Report ──────────────────────────────────────────────────────────


def print_final_report(
    records: list[ImageRecord],
    counts: dict[str, dict[str, int]],
    corrupted: list[ImageRecord],
    dup_groups: dict[str, list[ImageRecord]],
) -> None:
    _section("FINAL EDA REPORT — Risks & Recommendations")

    risks: list[str] = []
    recommendations: list[str] = []

    # Class imbalance
    train_c = counts.get("train", {})
    if train_c:
        ratio = train_c.get("PNEUMONIA", 0) / max(train_c.get("NORMAL", 1), 1)
        if ratio > 2.0:
            risks.append(f"Severe class imbalance in train set (ratio {ratio:.1f}:1 PNEUMONIA:NORMAL)")
            recommendations.append("Use weighted CrossEntropyLoss or oversampling for minority class")

    # Tiny validation set
    val_total = sum(counts.get("val", {}).values())
    if val_total < 50:
        risks.append(f"Validation set extremely small ({val_total} images)")
        recommendations.append("Validation metrics will be noisy — rely on test set for evaluation")

    # Corrupted files
    if corrupted:
        risks.append(f"{len(corrupted)} corrupted image(s) detected")
        recommendations.append("Remove or replace corrupted files before training")

    # Duplicates
    if dup_groups:
        total_dup = sum(len(recs) - 1 for recs in dup_groups.values())
        risks.append(f"{total_dup} exact duplicate image(s) across {len(dup_groups)} group(s)")
        recommendations.append("Review duplicates — may inflate metrics if across splits")

    # Variable dimensions
    valid = [r for r in records if not r.corrupted]
    unique_sizes = set((r.width, r.height) for r in valid)
    if len(unique_sizes) > 5:
        risks.append(f"Inconsistent image sizes ({len(unique_sizes)} unique dimensions)")
        recommendations.append("Resize + CenterCrop in preprocessing is essential")

    # Dark images
    very_dark = sum(1 for r in valid if r.brightness < 30)
    if very_dark > 0:
        risks.append(f"{very_dark} very dark image(s) (brightness < 30)")
        recommendations.append("Consider brightness normalization or CLAHE")

    # Report
    if risks:
        print("  RISKS:\n")
        for i, r in enumerate(risks, 1):
            print(f"    {i}. {r}")
        print("\n  RECOMMENDATIONS:\n")
        for i, r in enumerate(recommendations, 1):
            print(f"    {i}. {r}")
    else:
        print("  No significant risks detected. Dataset appears clean.")

    print(f"\n  Total images analysed: {len(records):,}")
    print(f"  Corrupted: {len(corrupted)}")
    print(f"  Output directory: {OUTPUT_DIR}")


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chest X-Ray Dataset — Professional Medical Imaging EDA",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--no-hash", action="store_true", help="Skip SHA-256 hashing (faster)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Chest X-Ray Dataset — Medical Imaging EDA")
    print("=" * 60)
    print(f"  Dataset: {DATA_DIR}")
    print(f"  Output:  {OUTPUT_DIR}")

    if not DATA_DIR.exists():
        print(f"\n  ERROR: Dataset not found: {DATA_DIR}")
        sys.exit(1)

    # ── Scan all images ───────────────────────────────────────────────
    _section("Scanning dataset...")

    start = time.time()
    records: list[ImageRecord] = []

    for split_name, split_dir in SPLITS.items():
        for class_name, file_path in _iter_images(split_dir):
            records.append(ImageRecord(split_name, class_name, file_path))

    print(f"  Found {len(records):,} image files")
    print(f"  Analysing (dimensions, brightness, contrast, sharpness{', SHA-256' if not args.no_hash else ''})...")

    for i, r in enumerate(records):
        r.analyse(compute_hash=not args.no_hash)
        if (i + 1) % 500 == 0:
            print(f"    ... {i + 1:,}/{len(records):,}")

    elapsed = time.time() - start
    print(f"  Scan complete in {elapsed:.1f}s")

    # ── Run all analysis sections ─────────────────────────────────────
    counts = run_inventory(records)
    corrupted = run_corrupted(records)
    run_dimensions(records)
    run_aspect_ratios(records)
    run_file_sizes(records)
    run_brightness_contrast(records)
    run_sharpness(records)
    run_duplicate_filenames(records)
    dup_groups = run_exact_duplicates(records)
    run_leakage_detection(records)
    run_class_comparison(records)

    # ── Plots ─────────────────────────────────────────────────────────
    if not args.no_plots:
        _section("Generating Plots")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        plot_class_distribution(counts)
        plot_dimension_scatter(records)
        plot_brightness_histograms(records)
        plot_contrast_comparison(records)
        plot_file_size_distribution(records)
        plot_sharpness_distribution(records)
        plot_sample_grid(records)

    # ── CSV Export ────────────────────────────────────────────────────
    _section("Exporting CSV")
    export_csv(records)

    # ── Final Report ──────────────────────────────────────────────────
    print_final_report(records, counts, corrupted, dup_groups)

    print("\n" + "=" * 60)
    print("  EDA complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Chest X-Ray Dataset — Medical Imaging EDA.

Complete exploratory data analysis for the chest_xray dataset.
Dependencies: PIL (Pillow), numpy. Optional: matplotlib for plots.

Usage:
    cd hms-ai
    python -m ml.eda.chest_xray_eda
    python -m ml.eda.chest_xray_eda --no-plots
    python -m ml.eda.chest_xray_eda --no-hash
    python -m ml.eda.chest_xray_eda --no-plots --no-hash
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from collections import defaultdict
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


def _heading(title: str) -> None:
    print(f"\n{'━' * 64}")
    print(f"  {title}")
    print(f"{'━' * 64}\n")


def _iter_images(split_dir: Path):
    """Yield (class_name, file_path) for every image in a split."""
    for cls in CLASS_NAMES:
        cdir = split_dir / cls
        if not cdir.exists():
            continue
        for f in sorted(cdir.iterdir()):
            if f.suffix.lower() in IMAGE_EXTENSIONS:
                yield cls, f


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_plt():
    """Return matplotlib.pyplot or None if unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


# ── Per-image record ──────────────────────────────────────────────────────


class Rec:
    """Stores all per-image statistics."""

    __slots__ = (
        "split", "cls", "path", "w", "h", "ar", "size_kb", "mode",
        "brightness", "contrast", "sharpness", "corrupted", "md5",
    )

    def __init__(self, split: str, cls: str, path: Path):
        self.split = split
        self.cls = cls
        self.path = path
        self.w: int = 0
        self.h: int = 0
        self.ar: float = 0.0
        self.size_kb: float = path.stat().st_size / 1024.0
        self.mode: str = ""
        self.brightness: float = 0.0
        self.contrast: float = 0.0
        self.sharpness: float = 0.0
        self.corrupted: bool = False
        self.md5: str = ""

    def analyse(self, do_hash: bool) -> None:
        try:
            img = Image.open(self.path)
            img.load()
        except Exception:
            self.corrupted = True
            return

        self.w, self.h = img.size
        self.ar = round(self.w / max(self.h, 1), 4)
        self.mode = img.mode

        gray = img.convert("L")
        stat = ImageStat.Stat(gray)
        self.brightness = round(stat.mean[0], 2)
        self.contrast = round(stat.stddev[0], 2)

        # Sharpness: variance of Laplacian (PIL edge filter, no scipy needed)
        edges = gray.filter(ImageFilter.Kernel(
            size=(3, 3),
            kernel=[0, 1, 0, 1, -4, 1, 0, 1, 0],
            scale=1,
            offset=128,
        ))
        self.sharpness = round(float(ImageStat.Stat(edges).stddev[0]), 2)

        img.close()
        if do_hash:
            self.md5 = _md5(self.path)


# ══════════════════════════════════════════════════════════════════════════
#  Analysis sections
# ══════════════════════════════════════════════════════════════════════════


def sec_inventory(recs: list[Rec]) -> dict[str, dict[str, int]]:
    _heading("1 · Dataset Inventory")

    counts: dict[str, dict[str, int]] = {s: {c: 0 for c in CLASS_NAMES} for s in SPLITS}
    for r in recs:
        counts[r.split][r.cls] += 1

    grand = 0
    for s in SPLITS:
        c = counts[s]
        tot = sum(c.values())
        grand += tot
        ratio = c["PNEUMONIA"] / max(c["NORMAL"], 1)
        pct = c["PNEUMONIA"] / max(tot, 1) * 100
        print(
            f"  {s:6s}  NORMAL={c['NORMAL']:>5}  PNEUMONIA={c['PNEUMONIA']:>5}  "
            f"Total={tot:>5}  Ratio={ratio:.2f}:1  Pneumonia%={pct:.1f}%"
        )
    print(f"\n  Grand total: {grand:,} images")
    return counts


def sec_corrupted(recs: list[Rec]) -> list[Rec]:
    _heading("2 · Corrupted Image Detection")
    bad = [r for r in recs if r.corrupted]
    if bad:
        print(f"  Found {len(bad)} corrupted file(s):")
        for r in bad:
            print(f"    [{r.split}/{r.cls}] {r.path.name}")
    else:
        print("  All files OK — no corruption detected.")
    return bad


def sec_dimensions(recs: list[Rec]) -> None:
    _heading("3 · Image Dimensions")
    ok = [r for r in recs if not r.corrupted]
    for s in SPLITS:
        sub = [r for r in ok if r.split == s]
        if not sub:
            continue
        ws = [r.w for r in sub]
        hs = [r.h for r in sub]
        print(
            f"  {s:6s}  W: {min(ws):>5}–{max(ws):<5} avg={int(np.mean(ws)):>5}  "
            f"H: {min(hs):>5}–{max(hs):<5} avg={int(np.mean(hs)):>5}  n={len(sub)}"
        )

    uniq = set((r.w, r.h) for r in ok)
    print(f"\n  Unique (W×H) combinations: {len(uniq)}")
    if len(uniq) <= 12:
        for w, h in sorted(uniq):
            n = sum(1 for r in ok if r.w == w and r.h == h)
            print(f"    {w}×{h}: {n}")


def sec_aspect_ratios(recs: list[Rec]) -> None:
    _heading("4 · Aspect Ratio Analysis")
    ok = [r for r in recs if not r.corrupted]
    ars = [r.ar for r in ok]
    sq = sum(1 for a in ars if 0.95 <= a <= 1.05)
    ls = sum(1 for a in ars if a > 1.05)
    pt = sum(1 for a in ars if a < 0.95)
    print(f"  Range:     {min(ars):.3f} – {max(ars):.3f}")
    print(f"  Mean:      {np.mean(ars):.3f}   Median: {np.median(ars):.3f}")
    print(f"  Square:    {sq:>5}  ({sq / len(ars) * 100:.1f}%)")
    print(f"  Landscape: {ls:>5}  ({ls / len(ars) * 100:.1f}%)")
    print(f"  Portrait:  {pt:>5}  ({pt / len(ars) * 100:.1f}%)")

    outliers = [r for r in ok if r.ar > 2.5 or r.ar < 0.5]
    if outliers:
        print(f"\n  Extreme aspect ratios ({len(outliers)}):")
        for r in outliers[:8]:
            print(f"    [{r.split}/{r.cls}] {r.path.name}  {r.w}×{r.h}  ar={r.ar:.3f}")


def sec_file_sizes(recs: list[Rec]) -> None:
    _heading("5 · File Size Analysis")
    ok = [r for r in recs if not r.corrupted]

    for s in SPLITS:
        sub = [r for r in ok if r.split == s]
        if not sub:
            continue
        sizes = [r.size_kb for r in sub]
        print(
            f"  {s:6s}  Min={min(sizes):>7.1f} KB  Max={max(sizes):>8.1f} KB  "
            f"Avg={np.mean(sizes):>7.1f} KB  Total={sum(sizes) / 1024:>7.1f} MB"
        )

    print(f"\n  Per class:")
    for cls in CLASS_NAMES:
        sub = [r for r in ok if r.cls == cls]
        sizes = [r.size_kb for r in sub]
        print(
            f"    {cls:12s}  Avg={np.mean(sizes):>7.1f} KB  "
            f"Median={np.median(sizes):>7.1f} KB  n={len(sub)}"
        )

    all_sz = [r.size_kb for r in ok]
    print(f"\n  Dataset total: {sum(all_sz) / 1024 / 1024:.2f} GB")

    tiny = [r for r in ok if r.size_kb < 5]
    huge = [r for r in ok if r.size_kb > 1500]
    if tiny:
        print(f"\n  ⚠ {len(tiny)} file(s) < 5 KB:")
        for r in tiny[:5]:
            print(f"    [{r.split}/{r.cls}] {r.path.name} ({r.size_kb:.1f} KB)")
    if huge:
        print(f"\n  Note: {len(huge)} file(s) > 1.5 MB")


def sec_brightness_contrast(recs: list[Rec]) -> None:
    _heading("6 · Brightness & Contrast")
    ok = [r for r in recs if not r.corrupted]

    print("  By class:")
    for cls in CLASS_NAMES:
        sub = [r for r in ok if r.cls == cls]
        bs = [r.brightness for r in sub]
        cs = [r.contrast for r in sub]
        print(
            f"    {cls:12s}  Brightness: mean={np.mean(bs):>6.1f}  std={np.std(bs):>5.1f}  "
            f"[{min(bs):.0f}–{max(bs):.0f}]"
        )
        print(
            f"    {'':12s}  Contrast:   mean={np.mean(cs):>6.1f}  std={np.std(cs):>5.1f}  "
            f"[{min(cs):.0f}–{max(cs):.0f}]"
        )

    print("\n  By split:")
    for s in SPLITS:
        sub = [r for r in ok if r.split == s]
        if not sub:
            continue
        bs = [r.brightness for r in sub]
        print(
            f"    {s:6s}  Brightness: mean={np.mean(bs):>6.1f}  "
            f"range=[{min(bs):.0f}–{max(bs):.0f}]  n={len(sub)}"
        )

    dark = [r for r in ok if r.brightness < 30]
    bright = [r for r in ok if r.brightness > 220]
    if dark:
        print(f"\n  ⚠ {len(dark)} very dark image(s) (brightness < 30)")
    if bright:
        print(f"  ⚠ {len(bright)} very bright image(s) (brightness > 220)")
        for r in bright[:5]:
            print(f"    [{r.split}/{r.cls}] {r.path.name}  brightness={r.brightness:.0f}")
    if not dark and not bright:
        print("\n  No brightness outliers detected.")


def sec_sharpness(recs: list[Rec]) -> None:
    _heading("7 · Sharpness / Blur Estimation")
    ok = [r for r in recs if not r.corrupted]

    for cls in CLASS_NAMES:
        sub = [r for r in ok if r.cls == cls]
        ss = [r.sharpness for r in sub]
        print(
            f"  {cls:12s}  mean={np.mean(ss):>6.1f}  "
            f"median={np.median(ss):>6.1f}  "
            f"min={min(ss):>5.1f}  max={max(ss):>6.1f}"
        )

    all_s = [r.sharpness for r in ok]
    p5 = np.percentile(all_s, 5)
    blurry = [r for r in ok if r.sharpness < p5]
    print(f"\n  5th percentile: {p5:.1f}")
    print(f"  Potentially blurry (below 5th pctile): {len(blurry)}")
    if blurry:
        for r in sorted(blurry, key=lambda x: x.sharpness)[:5]:
            print(f"    [{r.split}/{r.cls}] {r.path.name}  sharpness={r.sharpness:.1f}")


def sec_dup_filenames(recs: list[Rec]) -> None:
    _heading("8 · Duplicate Filename Detection")
    name_map: dict[str, list[Rec]] = defaultdict(list)
    for r in recs:
        name_map[r.path.name].append(r)

    dupes = {n: rs for n, rs in name_map.items() if len(rs) > 1}
    if not dupes:
        print("  No duplicate filenames across the dataset.")
    else:
        print(f"  {len(dupes)} filename(s) appear in multiple locations:")
        for n, rs in sorted(dupes.items())[:15]:
            locs = ", ".join(f"{r.split}/{r.cls}" for r in rs)
            print(f"    {n} → {locs}")
        if len(dupes) > 15:
            print(f"    … and {len(dupes) - 15} more")


def sec_exact_dups(recs: list[Rec]) -> dict[str, list[Rec]]:
    _heading("9 · Exact Duplicate Detection (MD5)")
    hashed = [r for r in recs if r.md5 and not r.corrupted]
    if not hashed:
        print("  Skipped — run without --no-hash to enable.")
        return {}

    hmap: dict[str, list[Rec]] = defaultdict(list)
    for r in hashed:
        hmap[r.md5].append(r)

    dupes = {h: rs for h, rs in hmap.items() if len(rs) > 1}
    redundant = sum(len(rs) - 1 for rs in dupes.values())

    if not dupes:
        print("  No exact duplicate images found.")
    else:
        print(f"  {len(dupes)} duplicate group(s), {redundant} redundant image(s):\n")
        for h, rs in sorted(dupes.items(), key=lambda x: -len(x[1]))[:12]:
            first = rs[0]
            print(f"    [{len(rs)}×] {first.split}/{first.cls}/{first.path.name}")
            for r in rs[1:]:
                print(f"         = {r.split}/{r.cls}/{r.path.name}")
        if len(dupes) > 12:
            print(f"    … and {len(dupes) - 12} more groups")
    return dupes


def sec_leakage(recs: list[Rec]) -> None:
    _heading("10 · Data Leakage Detection (Train ↔ Val ↔ Test)")
    hashed = [r for r in recs if r.md5 and not r.corrupted]
    if not hashed:
        print("  Skipped — run without --no-hash to enable.")
        return

    split_h: dict[str, set[str]] = defaultdict(set)
    h2r: dict[str, Rec] = {}
    for r in hashed:
        split_h[r.split].add(r.md5)
        h2r[f"{r.split}:{r.md5}"] = r

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    leaked = False
    for s1, s2 in pairs:
        overlap = split_h.get(s1, set()) & split_h.get(s2, set())
        if overlap:
            leaked = True
            print(f"  ⚠ LEAKAGE: {len(overlap)} identical image(s) in {s1} AND {s2}:")
            for h in sorted(overlap)[:8]:
                r1 = h2r.get(f"{s1}:{h}")
                r2 = h2r.get(f"{s2}:{h}")
                n1 = f"{r1.cls}/{r1.path.name}" if r1 else "?"
                n2 = f"{r2.cls}/{r2.path.name}" if r2 else "?"
                print(f"    {s1}/{n1}  ==  {s2}/{n2}")
        else:
            print(f"  {s1} ↔ {s2}: clean — no overlap.")

    if not leaked:
        print("\n  ✓ No cross-split leakage detected.")


def sec_class_comparison(recs: list[Rec]) -> None:
    _heading("11 · Class Comparison: NORMAL vs PNEUMONIA")
    ok = [r for r in recs if not r.corrupted]

    for cls in CLASS_NAMES:
        sub = [r for r in ok if r.cls == cls]
        bs = [r.brightness for r in sub]
        cs = [r.contrast for r in sub]
        ss = [r.sharpness for r in sub]
        fs = [r.size_kb for r in sub]
        ws = [r.w for r in sub]
        hs = [r.h for r in sub]

        print(f"  {cls} (n={len(sub)})")
        print(f"    Brightness   mean={np.mean(bs):>6.1f}  median={np.median(bs):>6.1f}  std={np.std(bs):>5.1f}")
        print(f"    Contrast     mean={np.mean(cs):>6.1f}  median={np.median(cs):>6.1f}  std={np.std(cs):>5.1f}")
        print(f"    Sharpness    mean={np.mean(ss):>6.1f}  median={np.median(ss):>6.1f}")
        print(f"    File size    mean={np.mean(fs):>7.1f} KB  median={np.median(fs):>7.1f} KB")
        print(f"    Width        mean={np.mean(ws):>7.0f} px  median={np.median(ws):>7.0f} px")
        print(f"    Height       mean={np.mean(hs):>7.0f} px  median={np.median(hs):>7.0f} px")
        print()


# ══════════════════════════════════════════════════════════════════════════
#  Plots
# ══════════════════════════════════════════════════════════════════════════


def _bar_label(ax, bars) -> None:
    for b in bars:
        ax.text(
            b.get_x() + b.get_width() / 2, b.get_height() + 20,
            str(int(b.get_height())), ha="center", va="bottom", fontsize=8,
        )


def plot_class_dist(counts: dict[str, dict[str, int]]) -> None:
    plt = _get_plt()
    if not plt:
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = list(counts.keys())
    x = np.arange(len(splits))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w / 2, [counts[s]["NORMAL"] for s in splits], w, label="NORMAL", color="#4CAF50")
    b2 = ax.bar(x + w / 2, [counts[s]["PNEUMONIA"] for s in splits], w, label="PNEUMONIA", color="#F44336")
    _bar_label(ax, b1)
    _bar_label(ax, b2)
    ax.set_xlabel("Split")
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution by Split")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=150)
    plt.close()
    print(f"  ✓ class_distribution.png")


def plot_dimensions(recs: list[Rec]) -> None:
    plt = _get_plt()
    if not plt:
        return
    ok = [r for r in recs if not r.corrupted]
    fig, ax = plt.subplots(figsize=(8, 6))
    for cls, color in [("NORMAL", "#4CAF50"), ("PNEUMONIA", "#F44336")]:
        sub = [r for r in ok if r.cls == cls]
        ax.scatter([r.w for r in sub], [r.h for r in sub], alpha=0.25, s=6, label=cls, color=color)
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title("Image Dimensions by Class")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dimension_scatter.png", dpi=150)
    plt.close()
    print(f"  ✓ dimension_scatter.png")


def plot_brightness(recs: list[Rec]) -> None:
    plt = _get_plt()
    if not plt:
        return
    ok = [r for r in recs if not r.corrupted]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, cls in enumerate(CLASS_NAMES):
        sub = [r for r in ok if r.cls == cls]
        vals = [r.brightness for r in sub]
        c = "#4CAF50" if cls == "NORMAL" else "#F44336"
        axes[i].hist(vals, bins=40, color=c, alpha=0.85, edgecolor="white")
        axes[i].axvline(np.mean(vals), color="black", ls="--", label=f"Mean={np.mean(vals):.0f}")
        axes[i].set_title(f"{cls} — Brightness")
        axes[i].set_xlabel("Mean Pixel Intensity")
        axes[i].set_ylabel("Count")
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "brightness_histograms.png", dpi=150)
    plt.close()
    print(f"  ✓ brightness_histograms.png")


def plot_contrast(recs: list[Rec]) -> None:
    plt = _get_plt()
    if not plt:
        return
    ok = [r for r in recs if not r.corrupted]
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [[r.contrast for r in ok if r.cls == cls] for cls in CLASS_NAMES]
    colors = ["#4CAF50", "#F44336"]
    bp = ax.boxplot(data, labels=CLASS_NAMES, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_ylabel("Contrast (Pixel Intensity Std Dev)")
    ax.set_title("Contrast: NORMAL vs PNEUMONIA")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "contrast_boxplot.png", dpi=150)
    plt.close()
    print(f"  ✓ contrast_boxplot.png")


def plot_file_sizes(recs: list[Rec]) -> None:
    plt = _get_plt()
    if not plt:
        return
    ok = [r for r in recs if not r.corrupted]
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls, c in [("NORMAL", "#4CAF50"), ("PNEUMONIA", "#F44336")]:
        vals = [r.size_kb for r in ok if r.cls == cls]
        ax.hist(vals, bins=50, alpha=0.55, label=cls, color=c, edgecolor="white")
    ax.set_xlabel("File Size (KB)")
    ax.set_ylabel("Count")
    ax.set_title("File Size Distribution by Class")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "file_size_distribution.png", dpi=150)
    plt.close()
    print(f"  ✓ file_size_distribution.png")


def plot_sharpness(recs: list[Rec]) -> None:
    plt = _get_plt()
    if not plt:
        return
    ok = [r for r in recs if not r.corrupted]
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls, c in [("NORMAL", "#4CAF50"), ("PNEUMONIA", "#F44336")]:
        vals = [r.sharpness for r in ok if r.cls == cls]
        p99 = np.percentile(vals, 99)
        clipped = [min(v, p99) for v in vals]
        ax.hist(clipped, bins=40, alpha=0.55, label=cls, color=c, edgecolor="white")
    ax.set_xlabel("Sharpness (Laplacian Variance)")
    ax.set_ylabel("Count")
    ax.set_title("Sharpness Distribution by Class")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sharpness_distribution.png", dpi=150)
    plt.close()
    print(f"  ✓ sharpness_distribution.png")


def plot_intensity_hist(recs: list[Rec]) -> None:
    """Pixel intensity histogram from a random sample of images."""
    plt = _get_plt()
    if not plt:
        return
    ok = [r for r in recs if not r.corrupted and r.split == "train"]

    rng = np.random.RandomState(42)
    sample_n = min(200, len(ok))
    indices = rng.choice(len(ok), sample_n, replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, cls in enumerate(CLASS_NAMES):
        sampled = [ok[j] for j in indices if ok[j].cls == cls][:100]
        all_pixels: list[np.ndarray] = []
        for r in sampled:
            try:
                img = Image.open(r.path).convert("L")
                all_pixels.append(np.array(img).ravel())
                img.close()
            except Exception:
                pass
        if all_pixels:
            merged = np.concatenate(all_pixels)
            c = "#4CAF50" if cls == "NORMAL" else "#F44336"
            axes[i].hist(merged, bins=64, color=c, alpha=0.85, edgecolor="white", density=True)
            axes[i].set_title(f"{cls} — Pixel Intensity (sample)")
            axes[i].set_xlabel("Pixel Value (0–255)")
            axes[i].set_ylabel("Density")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pixel_intensity_histograms.png", dpi=150)
    plt.close()
    print(f"  ✓ pixel_intensity_histograms.png")


def plot_sample_grid(recs: list[Rec]) -> None:
    plt = _get_plt()
    if not plt:
        return
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    fig.suptitle("Sample Chest X-Ray Images (Train Set)", fontsize=14)

    train = [r for r in recs if r.split == "train" and not r.corrupted]
    for row, cls in enumerate(CLASS_NAMES):
        sub = [r for r in train if r.cls == cls][:5]
        for col, r in enumerate(sub):
            try:
                img = Image.open(r.path).convert("L")
                axes[row, col].imshow(np.array(img), cmap="gray")
                axes[row, col].set_title(f"{cls}\n{r.w}×{r.h}", fontsize=8)
                img.close()
            except Exception:
                pass
            axes[row, col].axis("off")
        for col in range(len(sub), 5):
            axes[row, col].axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sample_grid.png", dpi=150)
    plt.close()
    print(f"  ✓ sample_grid.png")


def plot_aspect_ratio(recs: list[Rec]) -> None:
    plt = _get_plt()
    if not plt:
        return
    ok = [r for r in recs if not r.corrupted]
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls, c in [("NORMAL", "#4CAF50"), ("PNEUMONIA", "#F44336")]:
        vals = [r.ar for r in ok if r.cls == cls]
        ax.hist(vals, bins=40, alpha=0.55, label=cls, color=c, edgecolor="white")
    ax.set_xlabel("Aspect Ratio (W/H)")
    ax.set_ylabel("Count")
    ax.set_title("Aspect Ratio Distribution by Class")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "aspect_ratio_distribution.png", dpi=150)
    plt.close()
    print(f"  ✓ aspect_ratio_distribution.png")


# ══════════════════════════════════════════════════════════════════════════
#  CSV export
# ══════════════════════════════════════════════════════════════════════════


def export_csv(recs: list[Rec]) -> None:
    _heading("CSV Export")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "image_stats.csv"

    fields = [
        "split", "class", "filename", "width", "height", "aspect_ratio",
        "file_size_kb", "mode", "brightness", "contrast", "sharpness", "corrupted",
    ]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in recs:
            w.writerow({
                "split": r.split, "class": r.cls, "filename": r.path.name,
                "width": r.w, "height": r.h, "aspect_ratio": r.ar,
                "file_size_kb": round(r.size_kb, 2), "mode": r.mode,
                "brightness": r.brightness, "contrast": r.contrast,
                "sharpness": r.sharpness, "corrupted": r.corrupted,
            })
    print(f"  Saved {len(recs)} rows → {out}")


# ══════════════════════════════════════════════════════════════════════════
#  Final report
# ══════════════════════════════════════════════════════════════════════════


def final_report(
    recs: list[Rec],
    counts: dict[str, dict[str, int]],
    corrupted: list[Rec],
    dup_groups: dict[str, list[Rec]],
) -> None:
    _heading("FINAL REPORT — Risks & Recommendations")

    risks: list[str] = []
    recs_ok: list[str] = []

    # Class imbalance
    tc = counts.get("train", {})
    if tc:
        ratio = tc.get("PNEUMONIA", 0) / max(tc.get("NORMAL", 1), 1)
        if ratio > 2.0:
            risks.append(f"Severe class imbalance: train ratio {ratio:.1f}:1 (PNEUMONIA:NORMAL)")
            recs_ok.append("Use weighted CrossEntropyLoss or oversampling for NORMAL class")

    # Tiny val set
    val_n = sum(counts.get("val", {}).values())
    if val_n < 50:
        risks.append(f"Validation set critically small ({val_n} images)")
        recs_ok.append("Carve 10-15% from train as new validation split (stratified)")

    # Corrupted
    if corrupted:
        risks.append(f"{len(corrupted)} corrupted image(s)")
        recs_ok.append("Remove corrupted files before training")

    # Duplicates
    if dup_groups:
        n_dup = sum(len(rs) - 1 for rs in dup_groups.values())
        risks.append(f"{n_dup} exact duplicate image(s) in {len(dup_groups)} group(s)")
        recs_ok.append("Deduplicate training set to prevent metric inflation")

    # Dimension variation
    ok = [r for r in recs if not r.corrupted]
    uniq_w = len(set(r.w for r in ok))
    if uniq_w > 10:
        risks.append(f"Highly variable dimensions ({uniq_w} unique widths)")
        recs_ok.append("Resize(256) + CenterCrop(224) mandatory in preprocessing")

    # File size disparity
    n_sizes = [r.size_kb for r in ok if r.cls == "NORMAL"]
    p_sizes = [r.size_kb for r in ok if r.cls == "PNEUMONIA"]
    if n_sizes and p_sizes:
        ratio_fs = np.mean(n_sizes) / max(np.mean(p_sizes), 1)
        if ratio_fs > 3:
            risks.append(f"File size disparity: NORMAL avg {np.mean(n_sizes):.0f} KB vs PNEUMONIA {np.mean(p_sizes):.0f} KB ({ratio_fs:.1f}×)")
            recs_ok.append("Monitor for JPEG-artifact-based spurious correlation")

    # Mixed color modes
    rgb_n = sum(1 for r in ok if r.mode == "RGB")
    if rgb_n > 0:
        risks.append(f"{rgb_n} RGB images mixed with grayscale")
        recs_ok.append("Standardize all to RGB via .convert('RGB') for pretrained models")

    # Bright outliers
    bright = sum(1 for r in ok if r.brightness > 220)
    if bright:
        risks.append(f"{bright} overexposed image(s) (brightness > 220)")
        recs_ok.append("Review and potentially exclude overexposed images")

    # Print
    if risks:
        print("  RISKS:\n")
        for i, r in enumerate(risks, 1):
            print(f"    {i}. {r}")
        print("\n  RECOMMENDATIONS:\n")
        for i, r in enumerate(recs_ok, 1):
            print(f"    {i}. {r}")
    else:
        print("  No significant risks. Dataset appears clean.")

    print(f"\n  Images analysed:   {len(recs):,}")
    print(f"  Corrupted:         {len(corrupted)}")
    print(f"  Output directory:  {OUTPUT_DIR}")


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="Chest X-Ray EDA")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--no-hash", action="store_true", help="Skip MD5 hashing (faster)")
    args = parser.parse_args()

    print("=" * 64)
    print("  Chest X-Ray Dataset — Medical Imaging EDA")
    print("=" * 64)
    print(f"  Dataset: {DATA_DIR}")
    print(f"  Output:  {OUTPUT_DIR}")

    if not DATA_DIR.exists():
        print(f"\n  ERROR: Dataset not found at {DATA_DIR}")
        sys.exit(1)

    # ── Scan ──────────────────────────────────────────────────────────
    _heading("Scanning dataset")

    t0 = time.time()
    recs: list[Rec] = []
    for sname, sdir in SPLITS.items():
        for cls, fpath in _iter_images(sdir):
            recs.append(Rec(sname, cls, fpath))

    print(f"  Found {len(recs):,} images")
    hashing = not args.no_hash
    print(f"  Analysing (dimensions, brightness, contrast, sharpness{', MD5' if hashing else ''}) …")

    for i, r in enumerate(recs):
        r.analyse(do_hash=hashing)
        if (i + 1) % 500 == 0:
            print(f"    {i + 1:,} / {len(recs):,}")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # ── Analysis sections ─────────────────────────────────────────────
    counts = sec_inventory(recs)
    corrupted = sec_corrupted(recs)
    sec_dimensions(recs)
    sec_aspect_ratios(recs)
    sec_file_sizes(recs)
    sec_brightness_contrast(recs)
    sec_sharpness(recs)
    sec_dup_filenames(recs)
    dup_groups = sec_exact_dups(recs)
    sec_leakage(recs)
    sec_class_comparison(recs)

    # ── Plots ─────────────────────────────────────────────────────────
    if not args.no_plots:
        _heading("Generating Plots")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        plot_class_dist(counts)
        plot_dimensions(recs)
        plot_brightness(recs)
        plot_contrast(recs)
        plot_file_sizes(recs)
        plot_sharpness(recs)
        plot_intensity_hist(recs)
        plot_aspect_ratio(recs)
        plot_sample_grid(recs)

    # ── CSV ───────────────────────────────────────────────────────────
    export_csv(recs)

    # ── Final report ──────────────────────────────────────────────────
    final_report(recs, counts, corrupted, dup_groups)

    print("\n" + "=" * 64)
    print("  EDA complete.")
    print("=" * 64)


if __name__ == "__main__":
    main()

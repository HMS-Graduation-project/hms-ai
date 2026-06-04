"""
Chest X-Ray Dataset -- Medical Imaging EDA.

Complete exploratory data analysis for the chest_xray dataset.
Dependencies: PIL (Pillow), numpy. Optional: matplotlib, seaborn, scipy.

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
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageStat

# Resolve paths without importing training config (avoids torch dependency)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _PROJECT_ROOT / "app" / "data" / "chest_xray"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# ── Constants ─────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
PALETTE = {"NORMAL": "#4C72B0", "PNEUMONIA": "#DD8452"}
SPLITS: dict[str, Path] = {"train": TRAIN_DIR, "val": VAL_DIR, "test": TEST_DIR}
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
MEAN_IMG_SIZE = 256  # size for mean-image computation
PIXEL_SAMPLE_N = 300  # images to sample per class for pixel stats

# ── Helpers ───────────────────────────────────────────────────────────────


def _heading(title: str) -> None:
    print(f"\n{'=' * 64}")
    print(f"  {title}")
    print(f"{'=' * 64}\n")


def _iter_images(split_dir: Path):
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


def _avg_hash(path: Path, size: int = 16) -> int | None:
    """Perceptual average hash for near-duplicate detection."""
    try:
        img = Image.open(path).convert("L").resize((size, size), Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.float32)
        bits = (arr >= arr.mean()).flatten()
        return int(np.packbits(bits).tobytes().hex(), 16)
    except Exception:
        return None


def _get_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def _get_sns():
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid")
        return sns
    except ImportError:
        return None


# ── Per-image record ──────────────────────────────────────────────────────


class Rec:
    __slots__ = (
        "split", "cls", "path", "w", "h", "ar", "size_kb", "mode",
        "brightness", "contrast", "sharpness", "corrupted", "md5", "phash",
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
        self.phash: int | None = None

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

        edges = gray.filter(ImageFilter.Kernel(
            size=(3, 3), kernel=[0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1, offset=128,
        ))
        self.sharpness = round(float(ImageStat.Stat(edges).stddev[0]), 2)

        img.close()
        if do_hash:
            self.md5 = _md5(self.path)
            self.phash = _avg_hash(self.path)


# ══════════════════════════════════════════════════════════════════════════
#  Analysis sections
# ══════════════════════════════════════════════════════════════════════════


def sec_inventory(recs: list[Rec]) -> dict[str, dict[str, int]]:
    _heading("1 - Dataset Inventory")
    counts: dict[str, dict[str, int]] = {s: {c: 0 for c in CLASS_NAMES} for s in SPLITS}
    for r in recs:
        counts[r.split][r.cls] += 1
    grand = 0
    for s in SPLITS:
        c = counts[s]; tot = sum(c.values()); grand += tot
        ratio = c["PNEUMONIA"] / max(c["NORMAL"], 1)
        pct = c["PNEUMONIA"] / max(tot, 1) * 100
        print(f"  {s:6s}  NORMAL={c['NORMAL']:>5}  PNEUMONIA={c['PNEUMONIA']:>5}  "
              f"Total={tot:>5}  Ratio={ratio:.2f}:1  Pneumonia%={pct:.1f}%")
    print(f"\n  Grand total: {grand:,} images")
    return counts


def sec_corrupted(recs: list[Rec]) -> list[Rec]:
    _heading("2 - Corrupted Image Detection")
    bad = [r for r in recs if r.corrupted]
    if bad:
        print(f"  Found {len(bad)} corrupted file(s):")
        for r in bad:
            print(f"    [{r.split}/{r.cls}] {r.path.name}")
    else:
        print("  All files OK -- no corruption detected.")
    return bad


def sec_dimensions(recs: list[Rec]) -> None:
    _heading("3 - Image Dimensions")
    ok = [r for r in recs if not r.corrupted]
    for s in SPLITS:
        sub = [r for r in ok if r.split == s]
        if not sub: continue
        ws = [r.w for r in sub]; hs = [r.h for r in sub]
        print(f"  {s:6s}  W: {min(ws):>5}–{max(ws):<5} avg={int(np.mean(ws)):>5}  "
              f"H: {min(hs):>5}–{max(hs):<5} avg={int(np.mean(hs)):>5}  n={len(sub)}")
    uniq = set((r.w, r.h) for r in ok)
    print(f"\n  Unique (WxH) combinations: {len(uniq)}")
    modes = defaultdict(int)
    for r in ok: modes[r.mode] += 1
    print(f"  Image modes: {dict(modes)}")


def sec_aspect_ratios(recs: list[Rec]) -> None:
    _heading("4 - Aspect Ratio Analysis")
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
            print(f"    [{r.split}/{r.cls}] {r.path.name}  {r.w}x{r.h}  ar={r.ar:.3f}")


def sec_file_sizes(recs: list[Rec]) -> None:
    _heading("5 - File Size Analysis")
    ok = [r for r in recs if not r.corrupted]
    for s in SPLITS:
        sub = [r for r in ok if r.split == s]
        if not sub: continue
        sizes = [r.size_kb for r in sub]
        print(f"  {s:6s}  Min={min(sizes):>7.1f} KB  Max={max(sizes):>8.1f} KB  "
              f"Avg={np.mean(sizes):>7.1f} KB  Total={sum(sizes)/1024:>7.1f} MB")
    print(f"\n  Per class:")
    for cls in CLASS_NAMES:
        sub = [r for r in ok if r.cls == cls]; sizes = [r.size_kb for r in sub]
        print(f"    {cls:12s}  Avg={np.mean(sizes):>7.1f} KB  Median={np.median(sizes):>7.1f} KB  n={len(sub)}")
    all_sz = [r.size_kb for r in ok]
    print(f"\n  Dataset total: {sum(all_sz)/1024/1024:.2f} GB")
    tiny = [r for r in ok if r.size_kb < 5]
    if tiny:
        print(f"\n  WARNING: {len(tiny)} file(s) < 5 KB:")
        for r in tiny[:5]:
            print(f"    [{r.split}/{r.cls}] {r.path.name} ({r.size_kb:.1f} KB)")


def sec_brightness_contrast(recs: list[Rec]) -> None:
    _heading("6 - Brightness & Contrast")
    ok = [r for r in recs if not r.corrupted]
    print("  By class:")
    for cls in CLASS_NAMES:
        sub = [r for r in ok if r.cls == cls]
        bs = [r.brightness for r in sub]; cs = [r.contrast for r in sub]
        print(f"    {cls:12s}  Brightness: mean={np.mean(bs):>6.1f}  std={np.std(bs):>5.1f}  [{min(bs):.0f}–{max(bs):.0f}]")
        print(f"    {'':12s}  Contrast:   mean={np.mean(cs):>6.1f}  std={np.std(cs):>5.1f}  [{min(cs):.0f}–{max(cs):.0f}]")
    print("\n  By split:")
    for s in SPLITS:
        sub = [r for r in ok if r.split == s]
        if not sub: continue
        bs = [r.brightness for r in sub]
        print(f"    {s:6s}  Brightness: mean={np.mean(bs):>6.1f}  range=[{min(bs):.0f}–{max(bs):.0f}]  n={len(sub)}")
    dark = [r for r in ok if r.brightness < 30]
    bright = [r for r in ok if r.brightness > 220]
    if dark: print(f"\n  WARNING: {len(dark)} very dark image(s) (brightness < 30)")
    if bright:
        print(f"  WARNING: {len(bright)} very bright image(s) (brightness > 220)")
        for r in bright[:5]:
            print(f"    [{r.split}/{r.cls}] {r.path.name}  brightness={r.brightness:.0f}")
    if not dark and not bright:
        print("\n  No brightness outliers detected.")


def sec_sharpness(recs: list[Rec]) -> None:
    _heading("7 - Sharpness / Blur Estimation")
    ok = [r for r in recs if not r.corrupted]
    for cls in CLASS_NAMES:
        sub = [r for r in ok if r.cls == cls]; ss = [r.sharpness for r in sub]
        print(f"  {cls:12s}  mean={np.mean(ss):>6.1f}  median={np.median(ss):>6.1f}  "
              f"min={min(ss):>5.1f}  max={max(ss):>6.1f}")
    all_s = [r.sharpness for r in ok]; p5 = np.percentile(all_s, 5)
    blurry = [r for r in ok if r.sharpness < p5]
    print(f"\n  5th percentile: {p5:.1f}")
    print(f"  Potentially blurry (below 5th pctile): {len(blurry)}")
    if blurry:
        for r in sorted(blurry, key=lambda x: x.sharpness)[:5]:
            print(f"    [{r.split}/{r.cls}] {r.path.name}  sharpness={r.sharpness:.1f}")


def sec_dup_filenames(recs: list[Rec]) -> None:
    _heading("8 - Duplicate Filename Detection")
    name_map: dict[str, list[Rec]] = defaultdict(list)
    for r in recs: name_map[r.path.name].append(r)
    dupes = {n: rs for n, rs in name_map.items() if len(rs) > 1}
    if not dupes:
        print("  No duplicate filenames across the dataset.")
    else:
        print(f"  {len(dupes)} filename(s) appear in multiple locations:")
        for n, rs in sorted(dupes.items())[:15]:
            locs = ", ".join(f"{r.split}/{r.cls}" for r in rs)
            print(f"    {n} -> {locs}")


def sec_exact_dups(recs: list[Rec]) -> dict[str, list[Rec]]:
    _heading("9 - Exact Duplicate Detection (MD5)")
    hashed = [r for r in recs if r.md5 and not r.corrupted]
    if not hashed:
        print("  Skipped -- run without --no-hash to enable.")
        return {}
    hmap: dict[str, list[Rec]] = defaultdict(list)
    for r in hashed: hmap[r.md5].append(r)
    dupes = {h: rs for h, rs in hmap.items() if len(rs) > 1}
    redundant = sum(len(rs) - 1 for rs in dupes.values())
    if not dupes:
        print("  No exact duplicate images found.")
    else:
        print(f"  {len(dupes)} duplicate group(s), {redundant} redundant image(s):\n")
        for h, rs in sorted(dupes.items(), key=lambda x: -len(x[1]))[:12]:
            first = rs[0]
            print(f"    [{len(rs)}x] {first.split}/{first.cls}/{first.path.name}")
            for r in rs[1:]:
                print(f"         = {r.split}/{r.cls}/{r.path.name}")
    return dupes


def sec_near_dups(recs: list[Rec]) -> dict[int, list[Rec]]:
    """Near-duplicate detection via perceptual hash."""
    _heading("9b - Near-Duplicate Detection (Perceptual Hash)")
    hashed = [r for r in recs if r.phash is not None and not r.corrupted]
    if not hashed:
        print("  Skipped -- run without --no-hash to enable.")
        return {}
    pmap: dict[int, list[Rec]] = defaultdict(list)
    for r in hashed: pmap[r.phash].append(r)
    dupes = {h: rs for h, rs in pmap.items() if len(rs) > 1}
    redundant = sum(len(rs) - 1 for rs in dupes.values())
    if not dupes:
        print("  No perceptual near-duplicates found.")
    else:
        print(f"  {len(dupes)} group(s), {redundant} near-duplicate(s):")
        for h, rs in sorted(dupes.items(), key=lambda x: -len(x[1]))[:5]:
            names = [f"{r.split}/{r.cls}/{r.path.name}" for r in rs]
            print(f"    [{len(rs)}x] {names[0]}")
            for n in names[1:3]: print(f"         ~ {n}")
            if len(names) > 3: print(f"         ... +{len(names)-3} more")
    return dupes


def sec_leakage(recs: list[Rec]) -> None:
    _heading("10 - Data Leakage Detection (Train <-> Val <-> Test)")
    hashed = [r for r in recs if r.md5 and not r.corrupted]
    if not hashed:
        print("  Skipped -- run without --no-hash to enable.")
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
            print(f"  WARNING: LEAKAGE: {len(overlap)} identical image(s) in {s1} AND {s2}:")
            for h in sorted(overlap)[:8]:
                r1 = h2r.get(f"{s1}:{h}"); r2 = h2r.get(f"{s2}:{h}")
                n1 = f"{r1.cls}/{r1.path.name}" if r1 else "?"
                n2 = f"{r2.cls}/{r2.path.name}" if r2 else "?"
                print(f"    {s1}/{n1}  ==  {s2}/{n2}")
        else:
            print(f"  {s1} <-> {s2}: clean -- no overlap.")
    if not leaked:
        print("\n  OK: No cross-split leakage detected.")


def sec_class_comparison(recs: list[Rec]) -> dict[str, dict[str, float]]:
    _heading("11 - Class Comparison: NORMAL vs PNEUMONIA")
    ok = [r for r in recs if not r.corrupted]
    stats: dict[str, dict[str, float]] = {}
    for cls in CLASS_NAMES:
        sub = [r for r in ok if r.cls == cls]
        bs = [r.brightness for r in sub]; cs = [r.contrast for r in sub]
        ss = [r.sharpness for r in sub]; fs = [r.size_kb for r in sub]
        ws = [r.w for r in sub]; hs = [r.h for r in sub]
        stats[cls] = {
            "n": len(sub), "brightness_mean": np.mean(bs), "brightness_std": np.std(bs),
            "contrast_mean": np.mean(cs), "contrast_std": np.std(cs),
            "sharpness_mean": np.mean(ss), "file_size_mean": np.mean(fs),
            "width_mean": np.mean(ws), "height_mean": np.mean(hs),
        }
        print(f"  {cls} (n={len(sub)})")
        print(f"    Brightness   mean={np.mean(bs):>6.1f}  median={np.median(bs):>6.1f}  std={np.std(bs):>5.1f}")
        print(f"    Contrast     mean={np.mean(cs):>6.1f}  median={np.median(cs):>6.1f}  std={np.std(cs):>5.1f}")
        print(f"    Sharpness    mean={np.mean(ss):>6.1f}  median={np.median(ss):>6.1f}")
        print(f"    File size    mean={np.mean(fs):>7.1f} KB  median={np.median(fs):>7.1f} KB")
        print(f"    Width        mean={np.mean(ws):>7.0f} px   Height mean={np.mean(hs):>7.0f} px")
        print()
    return stats


def sec_class_weights(counts: dict[str, dict[str, int]]) -> dict[int, float]:
    """Compute sklearn-compatible balanced class weights."""
    _heading("12 - Class Weights (for Training)")
    tc = counts.get("train", {})
    n = tc.get("NORMAL", 0); p = tc.get("PNEUMONIA", 0); total = n + p
    if total == 0:
        print("  No training data."); return {}
    # balanced: n_samples / (n_classes * n_samples_class)
    w0 = total / (2 * max(n, 1))
    w1 = total / (2 * max(p, 1))
    print(f"  NORMAL    (0): weight = {w0:.4f}")
    print(f"  PNEUMONIA (1): weight = {w1:.4f}")
    print(f"  Ratio: NORMAL gets {w0/w1:.2f}x the gradient weight of PNEUMONIA")
    return {0: round(w0, 4), 1: round(w1, 4)}


# ══════════════════════════════════════════════════════════════════════════
#  Plots
# ══════════════════════════════════════════════════════════════════════════


def _save(plt, name: str) -> None:
    plt.savefig(OUTPUT_DIR / name, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  OK: {name}")


def plot_class_dist(counts: dict[str, dict[str, int]]) -> None:
    plt = _get_plt()
    if not plt: return
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Bar chart (train)
    tr_n = counts["train"]["NORMAL"]; tr_p = counts["train"]["PNEUMONIA"]
    bars = axes[0].bar(CLASS_NAMES, [tr_n, tr_p], color=list(PALETTE.values()), width=0.45, edgecolor="white")
    for bar, val in zip(bars, [tr_n, tr_p]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 40,
                     f"{val:,}", ha="center", fontweight="bold", fontsize=11)
    axes[0].set_title("Class Counts (Train)", fontweight="bold")
    axes[0].set_ylabel("Count"); axes[0].grid(axis="y", alpha=0.3)

    # Pie chart (train)
    axes[1].pie([tr_n, tr_p],
                labels=[f"NORMAL\n{100*tr_n/(tr_n+tr_p):.1f}%", f"PNEUMONIA\n{100*tr_p/(tr_n+tr_p):.1f}%"],
                colors=list(PALETTE.values()), startangle=90,
                textprops={"fontsize": 11}, wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1].set_title("Class Proportion (Train)", fontweight="bold")

    # Grouped bar across splits
    splits_l = list(counts.keys()); x = np.arange(len(splits_l)); w = 0.35
    axes[2].bar(x - w/2, [counts[s]["NORMAL"] for s in splits_l], w, label="NORMAL", color=PALETTE["NORMAL"])
    axes[2].bar(x + w/2, [counts[s]["PNEUMONIA"] for s in splits_l], w, label="PNEUMONIA", color=PALETTE["PNEUMONIA"])
    axes[2].set_xticks(x); axes[2].set_xticklabels(splits_l)
    axes[2].set_title("Per-Split Counts", fontweight="bold"); axes[2].legend(); axes[2].grid(axis="y", alpha=0.3)

    plt.suptitle("Class Distribution Analysis", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(plt, "class_distribution.png")


def plot_dimensions(recs: list[Rec]) -> None:
    plt = _get_plt()
    if not plt: return
    ok = [r for r in recs if not r.corrupted]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Width histogram
    for cls in CLASS_NAMES:
        vals = [r.w for r in ok if r.cls == cls]
        axes[0, 0].hist(vals, bins=50, alpha=0.6, color=PALETTE[cls], label=cls, edgecolor="none")
    axes[0, 0].set_title("Width Distribution", fontweight="bold"); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # Height histogram
    for cls in CLASS_NAMES:
        vals = [r.h for r in ok if r.cls == cls]
        axes[0, 1].hist(vals, bins=50, alpha=0.6, color=PALETTE[cls], label=cls, edgecolor="none")
    axes[0, 1].set_title("Height Distribution", fontweight="bold"); axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # Aspect ratio
    ars = [r.ar for r in ok]
    axes[1, 0].hist(ars, bins=50, color="#2ecc71", edgecolor="none", alpha=0.8)
    axes[1, 0].axvline(1.0, color="red", lw=2, ls="--", label="Square")
    axes[1, 0].axvline(np.median(ars), color="orange", lw=2, ls="--", label=f"Median={np.median(ars):.2f}")
    axes[1, 0].set_title("Aspect Ratio (W/H)", fontweight="bold"); axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    # Scatter
    rng = np.random.RandomState(42)
    for cls in CLASS_NAMES:
        sub = [r for r in ok if r.cls == cls]
        idx = rng.choice(len(sub), min(300, len(sub)), replace=False)
        axes[1, 1].scatter([sub[i].w for i in idx], [sub[i].h for i in idx],
                           alpha=0.35, s=12, label=cls, color=PALETTE[cls])
    axes[1, 1].set_title("Width vs Height", fontweight="bold"); axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    plt.suptitle("Image Dimension Analysis", fontsize=15, fontweight="bold")
    plt.tight_layout()
    _save(plt, "dimension_analysis.png")

    # Boxplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, attr, title in [(axes[0], "w", "Width"), (axes[1], "h", "Height")]:
        data = [[getattr(r, attr) for r in ok if r.cls == cls] for cls in CLASS_NAMES]
        bp = ax.boxplot(data, labels=CLASS_NAMES, patch_artist=True, medianprops=dict(color="white", lw=2))
        for patch, c in zip(bp["boxes"], PALETTE.values()):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        ax.set_title(f"{title} by Class", fontweight="bold"); ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Dimension Box Plots", fontweight="bold")
    plt.tight_layout()
    _save(plt, "dimension_boxplots.png")


def plot_brightness_contrast(recs: list[Rec]) -> None:
    plt = _get_plt()
    if not plt: return
    ok = [r for r in recs if not r.corrupted]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, cls in enumerate(CLASS_NAMES):
        vals = [r.brightness for r in ok if r.cls == cls]
        axes[i].hist(vals, bins=40, color=PALETTE[cls], alpha=0.85, edgecolor="white")
        axes[i].axvline(np.mean(vals), color="black", ls="--", label=f"Mean={np.mean(vals):.0f}")
        axes[i].set_title(f"{cls} -- Brightness", fontweight="bold"); axes[i].legend()
    plt.tight_layout()
    _save(plt, "brightness_histograms.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    data = [[r.contrast for r in ok if r.cls == cls] for cls in CLASS_NAMES]
    bp = ax.boxplot(data, labels=CLASS_NAMES, patch_artist=True)
    for patch, c in zip(bp["boxes"], PALETTE.values()):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.set_ylabel("Contrast (Std Dev)"); ax.set_title("Contrast: NORMAL vs PNEUMONIA", fontweight="bold")
    plt.tight_layout()
    _save(plt, "contrast_boxplot.png")


def plot_pixel_intensity(recs: list[Rec]) -> None:
    plt = _get_plt()
    if not plt: return
    ok = [r for r in recs if not r.corrupted and r.split == "train"]
    rng = np.random.RandomState(42)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    px_data: dict[str, np.ndarray] = {}

    for i, cls in enumerate(CLASS_NAMES):
        sub = [r for r in ok if r.cls == cls]
        idx = rng.choice(len(sub), min(PIXEL_SAMPLE_N, len(sub)), replace=False)
        all_px: list[np.ndarray] = []
        for j in idx:
            try:
                img = Image.open(sub[j].path).convert("L").resize((128, 128), Image.Resampling.LANCZOS)
                all_px.append(np.array(img, dtype=np.float32).ravel())
                img.close()
            except Exception: pass
        if all_px:
            merged = np.concatenate(all_px); px_data[cls] = merged
            axes[0].hist(merged, bins=128, range=(0, 255), density=True,
                         alpha=0.6, color=PALETTE[cls], label=cls, edgecolor="none")
    axes[0].set_title("Pixel Intensity Histogram", fontweight="bold")
    axes[0].set_xlabel("Pixel Value"); axes[0].legend(); axes[0].grid(alpha=0.3)

    # KDE curves
    try:
        from scipy.stats import gaussian_kde
        xs = np.linspace(0, 255, 512)
        for cls in CLASS_NAMES:
            if cls in px_data:
                sub_sample = rng.choice(px_data[cls], min(20000, len(px_data[cls])), replace=False)
                kde = gaussian_kde(sub_sample)
                axes[1].plot(xs, kde(xs), color=PALETTE[cls], lw=2.5, label=cls)
                axes[1].fill_between(xs, kde(xs), alpha=0.15, color=PALETTE[cls])
        axes[1].set_title("Pixel Intensity KDE", fontweight="bold")
        axes[1].set_xlabel("Pixel Value"); axes[1].legend(); axes[1].grid(alpha=0.3)
    except ImportError:
        axes[1].text(0.5, 0.5, "scipy not installed\n(KDE skipped)", transform=axes[1].transAxes, ha="center")

    # CDF
    for cls in CLASS_NAMES:
        if cls in px_data:
            sub_sample = rng.choice(px_data[cls], min(20000, len(px_data[cls])), replace=False)
            axes[2].plot(np.sort(sub_sample), np.linspace(0, 1, len(sub_sample)),
                         color=PALETTE[cls], lw=2.5, label=cls)
    axes[2].set_title("Cumulative Distribution", fontweight="bold")
    axes[2].set_xlabel("Pixel Value"); axes[2].set_ylabel("Proportion"); axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.suptitle("Pixel Intensity Analysis", fontsize=15, fontweight="bold")
    plt.tight_layout()
    _save(plt, "pixel_intensity.png")


def plot_sharpness_filesize(recs: list[Rec]) -> None:
    plt = _get_plt()
    if not plt: return
    ok = [r for r in recs if not r.corrupted]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for cls in CLASS_NAMES:
        vals = [r.sharpness for r in ok if r.cls == cls]
        p99 = np.percentile(vals, 99)
        axes[0].hist([min(v, p99) for v in vals], bins=40, alpha=0.55, label=cls, color=PALETTE[cls], edgecolor="white")
    axes[0].set_title("Sharpness Distribution", fontweight="bold"); axes[0].legend()

    for cls in CLASS_NAMES:
        vals = [r.size_kb for r in ok if r.cls == cls]
        axes[1].hist(vals, bins=50, alpha=0.55, label=cls, color=PALETTE[cls], edgecolor="white")
    axes[1].set_title("File Size Distribution", fontweight="bold"); axes[1].legend()

    plt.tight_layout()
    _save(plt, "sharpness_filesize.png")


def plot_mean_xrays(recs: list[Rec]) -> None:
    """Mean X-ray per class + difference heatmap."""
    plt = _get_plt()
    if not plt: return
    ok = [r for r in recs if not r.corrupted and r.split == "train"]
    rng = np.random.RandomState(42)

    mean_imgs: dict[str, np.ndarray] = {}
    for cls in CLASS_NAMES:
        sub = [r for r in ok if r.cls == cls]
        idx = rng.choice(len(sub), min(200, len(sub)), replace=False)
        accum = np.zeros((MEAN_IMG_SIZE, MEAN_IMG_SIZE), dtype=np.float64)
        count = 0
        for j in idx:
            try:
                arr = np.array(Image.open(sub[j].path).convert("L").resize(
                    (MEAN_IMG_SIZE, MEAN_IMG_SIZE), Image.Resampling.LANCZOS), dtype=np.float64)
                accum += arr; count += 1
            except Exception: pass
        mean_imgs[cls] = (accum / max(count, 1)).astype(np.float32)

    diff = mean_imgs["PNEUMONIA"] - mean_imgs["NORMAL"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes: ax.axis("off")

    im0 = axes[0].imshow(mean_imgs["NORMAL"], cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Mean NORMAL", color="white", fontweight="bold")

    im1 = axes[1].imshow(mean_imgs["PNEUMONIA"], cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Mean PNEUMONIA", color="white", fontweight="bold")

    im2 = axes[2].imshow(diff, cmap="RdBu_r", vmin=-40, vmax=40)
    axes[2].set_title("Difference\n(PNEUMONIA − NORMAL)", color="white", fontweight="bold")

    im3 = axes[3].imshow(np.abs(diff), cmap="hot", vmin=0, vmax=40)
    axes[3].set_title("Absolute Diff\n(Hotspots)", color="white", fontweight="bold")

    plt.suptitle("Mean X-Ray Appearance per Class", color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mean_xray_comparison.png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  OK: mean_xray_comparison.png")

    # Intensity profiles
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for cls in CLASS_NAMES:
        axes[0].plot(mean_imgs[cls].mean(axis=1), color=PALETTE[cls], lw=2.5, label=cls)
        axes[1].plot(mean_imgs[cls].mean(axis=0), color=PALETTE[cls], lw=2.5, label=cls)
    axes[0].set_title("Vertical Intensity Profile", fontweight="bold")
    axes[0].set_xlabel("Row (top->bottom)"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_title("Horizontal Intensity Profile", fontweight="bold")
    axes[1].set_xlabel("Column (left->right)"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.suptitle("Intensity Profiles: NORMAL vs PNEUMONIA", fontweight="bold")
    plt.tight_layout()
    _save(plt, "intensity_profiles.png")


def plot_sample_grids(recs: list[Rec]) -> None:
    plt = _get_plt()
    if not plt: return
    rng = np.random.RandomState(42)

    # Per-class grid
    for cls in CLASS_NAMES:
        sub = [r for r in recs if r.split == "train" and r.cls == cls and not r.corrupted]
        idx = rng.choice(len(sub), min(12, len(sub)), replace=False)
        rows, cols = 3, 4
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        fig.patch.set_facecolor("#1a1a2e")
        for i, ax in enumerate(axes.flat):
            if i < len(idx):
                try:
                    img = Image.open(sub[idx[i]].path).convert("L")
                    w, h = img.size
                    ax.imshow(np.array(img), cmap="gray", aspect="auto")
                    ax.set_title(f"{w}x{h}", fontsize=8, color="white",
                                 bbox=dict(boxstyle="round,pad=0.2", facecolor="#333", alpha=0.7))
                except Exception: pass
            ax.axis("off")
        plt.suptitle(f"{cls} -- Random Samples", fontsize=14, fontweight="bold", color="white")
        plt.tight_layout()
        fname = f"samples_{cls.lower()}.png"
        plt.savefig(OUTPUT_DIR / fname, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"  OK: {fname}")

    # Side-by-side comparison
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.patch.set_facecolor("#1a1a2e")
    for row, cls in enumerate(CLASS_NAMES):
        sub = [r for r in recs if r.split == "train" and r.cls == cls and not r.corrupted]
        idx = rng.choice(len(sub), min(4, len(sub)), replace=False)
        for col, j in enumerate(idx):
            try:
                img = Image.open(sub[j].path).convert("L")
                axes[row, col].imshow(np.array(img), cmap="gray")
            except Exception: pass
            axes[row, col].axis("off")
            axes[row, col].set_title(cls, fontsize=10, color=PALETTE[cls], fontweight="bold")
    plt.suptitle("NORMAL vs PNEUMONIA -- Side by Side", fontsize=14, fontweight="bold", color="white")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "normal_vs_pneumonia.png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  OK: normal_vs_pneumonia.png")


def plot_augmentation_preview(recs: list[Rec]) -> None:
    """Show what common augmentations look like on actual X-rays."""
    plt = _get_plt()
    if not plt: return
    try:
        from scipy import ndimage as ndi
    except ImportError:
        print("  [skip] scipy not installed -- augmentation preview skipped")
        return

    ok = [r for r in recs if r.split == "train" and not r.corrupted]

    def _load_sample(cls: str) -> np.ndarray:
        sub = [r for r in ok if r.cls == cls]
        return np.array(Image.open(sub[5].path).convert("L").resize((300, 300)), dtype=np.uint8)

    img_n = _load_sample("NORMAL"); img_p = _load_sample("PNEUMONIA")

    augs = [
        ("Original",       lambda x: x),
        ("Rotate +15°",    lambda x: ndi.rotate(x, 15, reshape=False, mode="nearest")),
        ("Rotate −15°",    lambda x: ndi.rotate(x, -15, reshape=False, mode="nearest")),
        ("Zoom 15%",       lambda x: np.array(Image.fromarray(
            x[int(x.shape[0]*.15):int(x.shape[0]*.85), int(x.shape[1]*.15):int(x.shape[1]*.85)]
        ).resize((x.shape[1], x.shape[0]), Image.Resampling.LANCZOS))),
        ("H-Flip",         lambda x: np.fliplr(x)),
        ("Shift",          lambda x: ndi.shift(x, [15, 20], mode="nearest")),
        ("Bright +30%",    lambda x: np.clip(x.astype(np.float32) * 1.3, 0, 255).astype(np.uint8)),
    ]

    fig, axes = plt.subplots(2, len(augs), figsize=(3 * len(augs), 7))
    fig.patch.set_facecolor("#1a1a2e")
    for col, (name, fn) in enumerate(augs):
        for row, (base, lbl) in enumerate([(img_n, "NORMAL"), (img_p, "PNEUMONIA")]):
            axes[row, col].imshow(fn(base), cmap="gray", aspect="auto")
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(name, fontsize=8.5, color="white", fontweight="bold")
    plt.suptitle("Data Augmentation Preview", fontsize=14, fontweight="bold", color="white")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "augmentation_preview.png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  OK: augmentation_preview.png")


# ══════════════════════════════════════════════════════════════════════════
#  Exports
# ══════════════════════════════════════════════════════════════════════════


def export_csv(recs: list[Rec]) -> None:
    _heading("CSV Export")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "image_stats.csv"
    fields = ["split", "class", "filename", "width", "height", "aspect_ratio",
              "file_size_kb", "mode", "brightness", "contrast", "sharpness", "corrupted"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in recs:
            w.writerow({"split": r.split, "class": r.cls, "filename": r.path.name,
                        "width": r.w, "height": r.h, "aspect_ratio": r.ar,
                        "file_size_kb": round(r.size_kb, 2), "mode": r.mode,
                        "brightness": r.brightness, "contrast": r.contrast,
                        "sharpness": r.sharpness, "corrupted": r.corrupted})
    print(f"  Saved {len(recs)} rows -> {out}")


def export_json(
    recs: list[Rec], counts: dict[str, dict[str, int]],
    class_stats: dict[str, dict[str, float]],
    class_weights: dict[int, float],
    dup_groups: dict[str, list[Rec]], near_dup_groups: dict[int, list[Rec]],
    corrupted: list[Rec],
) -> None:
    _heading("JSON Export")
    ok = [r for r in recs if not r.corrupted]
    total = sum(sum(c.values()) for c in counts.values())
    data = {
        "dataset_counts": {s: dict(c) for s, c in counts.items()},
        "totals": {cls: sum(counts[s][cls] for s in counts) for cls in CLASS_NAMES},
        "grand_total": total,
        "imbalance_ratio": round(counts["train"]["PNEUMONIA"] / max(counts["train"]["NORMAL"], 1), 4),
        "class_weights": {str(k): v for k, v in class_weights.items()},
        "class_stats": {cls: {k: round(v, 2) for k, v in s.items()} for cls, s in class_stats.items()},
        "quality": {
            "corrupted": len(corrupted),
            "unique_dimensions": len(set((r.w, r.h) for r in ok)),
            "rgb_images": sum(1 for r in ok if r.mode == "RGB"),
        },
        "duplicates": {
            "exact_groups": len(dup_groups),
            "exact_redundant": sum(len(rs) - 1 for rs in dup_groups.values()),
            "near_dup_groups": len(near_dup_groups),
        },
    }
    out = OUTPUT_DIR / "eda_data.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved -> {out}")


# ══════════════════════════════════════════════════════════════════════════
#  Final report
# ══════════════════════════════════════════════════════════════════════════


def final_report(
    recs: list[Rec], counts: dict[str, dict[str, int]],
    corrupted: list[Rec], dup_groups: dict[str, list[Rec]],
    near_dup_groups: dict[int, list[Rec]], class_weights: dict[int, float],
) -> None:
    _heading("FINAL REPORT -- Risks & Recommendations")
    risks: list[str] = []
    recs_ok: list[str] = []
    ok = [r for r in recs if not r.corrupted]
    tc = counts.get("train", {})

    ratio = tc.get("PNEUMONIA", 0) / max(tc.get("NORMAL", 1), 1)
    if ratio > 2.0:
        risks.append(f"Severe class imbalance: train ratio {ratio:.1f}:1 (PNEUMONIA:NORMAL)")
        w_str = f"NORMAL={class_weights.get(0, '?')}, PNEUMONIA={class_weights.get(1, '?')}"
        recs_ok.append(f"Use weighted loss: {{{w_str}}}")

    val_n = sum(counts.get("val", {}).values())
    if val_n < 50:
        risks.append(f"Validation set critically small ({val_n} images)")
        recs_ok.append("Carve 10-15% from train as stratified validation split")

    if corrupted:
        risks.append(f"{len(corrupted)} corrupted image(s)")
        recs_ok.append("Remove corrupted files before training")

    if dup_groups:
        n_dup = sum(len(rs) - 1 for rs in dup_groups.values())
        risks.append(f"{n_dup} exact duplicate(s) in {len(dup_groups)} group(s)")
        recs_ok.append("Deduplicate training set")

    if near_dup_groups:
        n_near = sum(len(rs) - 1 for rs in near_dup_groups.values())
        risks.append(f"{n_near} perceptual near-duplicate(s) in {len(near_dup_groups)} group(s)")
        recs_ok.append("Review near-duplicates for data quality")

    uniq_w = len(set(r.w for r in ok))
    if uniq_w > 10:
        risks.append(f"Highly variable dimensions ({uniq_w} unique widths)")
        recs_ok.append("Resize(256) + CenterCrop(224) mandatory")

    n_sizes = [r.size_kb for r in ok if r.cls == "NORMAL"]
    p_sizes = [r.size_kb for r in ok if r.cls == "PNEUMONIA"]
    if n_sizes and p_sizes:
        fs_ratio = np.mean(n_sizes) / max(np.mean(p_sizes), 1)
        if fs_ratio > 3:
            risks.append(f"File size disparity: NORMAL {np.mean(n_sizes):.0f} KB vs PNEUMONIA {np.mean(p_sizes):.0f} KB ({fs_ratio:.1f}x)")
            recs_ok.append("Monitor for JPEG-artifact spurious correlation")

    rgb_n = sum(1 for r in ok if r.mode == "RGB")
    if rgb_n > 0:
        risks.append(f"{rgb_n} RGB images mixed with grayscale")
        recs_ok.append("Standardize to RGB via .convert('RGB')")

    bright = sum(1 for r in ok if r.brightness > 220)
    if bright:
        risks.append(f"{bright} overexposed image(s) (brightness > 220)")
        recs_ok.append("Review overexposed images")

    if risks:
        print("  RISKS:\n")
        for i, r in enumerate(risks, 1): print(f"    {i}. {r}")
        print("\n  RECOMMENDATIONS:\n")
        for i, r in enumerate(recs_ok, 1): print(f"    {i}. {r}")
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
    parser.add_argument("--no-hash", action="store_true", help="Skip MD5/perceptual hashing")
    args = parser.parse_args()

    print("=" * 64)
    print("  Chest X-Ray Dataset -- Medical Imaging EDA")
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
    print(f"  Analysing (dimensions, brightness, contrast, sharpness"
          f"{', MD5, perceptual hash' if hashing else ''}) ...")

    for i, r in enumerate(recs):
        r.analyse(do_hash=hashing)
        if (i + 1) % 500 == 0:
            print(f"    {i + 1:,} / {len(recs):,}")

    print(f"  Done in {time.time() - t0:.1f}s")

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
    near_dup_groups = sec_near_dups(recs)
    sec_leakage(recs)
    class_stats = sec_class_comparison(recs)
    class_weights = sec_class_weights(counts)

    # ── Plots ─────────────────────────────────────────────────────────
    if not args.no_plots:
        _heading("Generating Plots")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        plot_class_dist(counts)
        plot_dimensions(recs)
        plot_brightness_contrast(recs)
        plot_pixel_intensity(recs)
        plot_sharpness_filesize(recs)
        plot_mean_xrays(recs)
        plot_sample_grids(recs)
        plot_augmentation_preview(recs)

    # ── Exports ───────────────────────────────────────────────────────
    export_csv(recs)
    export_json(recs, counts, class_stats, class_weights, dup_groups, near_dup_groups, corrupted)

    # ── Final report ──────────────────────────────────────────────────
    final_report(recs, counts, corrupted, dup_groups, near_dup_groups, class_weights)

    print("\n" + "=" * 64)
    print("  EDA complete.")
    print("=" * 64)


if __name__ == "__main__":
    main()

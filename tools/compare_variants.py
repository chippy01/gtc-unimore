"""
Plot the Image-GS runs of several comparisons of the same texture against each other, e.g. plain Gaussians (main
branch) against Beta Splatting (feat_beta_splatting), on the metrics and size axis of rd_curves.png.
Comparisons are paired by the BC/ASTC folder they were scored against, so every variant of one texture (other
kernels, bit precisions, -x options) lands on the same plot. Only metrics.csv files are read, no GPU is needed.
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Categorical palette and marker shapes of plot_compression.py, so color is never the only cue
PALETTE = [("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "D"), ("#eda100", "^"), ("#e87ba4", "v"),
           ("#008300", "<"), ("#6250d6", ">")]
METRICS = [("psnr", "PSNR (dB), higher is better"), ("ssim", "SSIM, higher is better"),
           ("lpips", "LPIPS, lower is better"), ("flip", "FLIP, lower is better")]
SIZES = {"file": ("bpp", "Size: the .gtc file as stored, like rd_curves.png"),
         "disk": ("zstd_bpp", "Size: the .gtc file compressed with Zstd, like the filled markers of sizes.png"),
         "gpu": ("gpu_bpp", "Size: the Gaussians plus the per-tile lists a shader needs, in GPU memory")}
SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def read_comparison(path):
    """Image-GS rows and baseline rows of one comparison folder, or None if it has no metrics.csv to compare."""
    csv_path = os.path.join(path, "metrics.csv")
    if not os.path.isfile(csv_path):
        return None
    with open(csv_path, "r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    variants = [row for row in rows if row["method"].startswith("Image-GS")]
    baselines = {(row["method"], row["setting"]): row for row in rows if not row["method"].startswith("Image-GS")}
    if not variants or not baselines:
        return None
    # non_gaussian_compressions/<texture>/<image>: the same texture, at the same cap, in every comparison
    texture_dir = os.path.dirname(os.path.abspath(next(iter(baselines.values()))["file"]))
    return {"path": path, "method": variants[0]["method"], "rows": variants, "baselines": baselines, "key": texture_dir}


def check_scoring(group):
    """Warn when the comparisons of one texture scored the same BC/ASTC files differently (other mask or options)."""
    first = group[0]
    for other in group[1:]:
        shared = set(first["baselines"]) & set(other["baselines"])
        worst = max((abs(float(first["baselines"][key]["psnr"]) - float(other["baselines"][key]["psnr"]))
                     for key in shared), default=0.0)
        if worst > 0.01:
            print(f"Warning: '{first['path']}' and '{other['path']}' score the same BC/ASTC files up to {worst:.2f} dB "
                  "apart, so they used another mask or options and their Image-GS runs are not comparable")


def count_label(setting):
    """'40000 Gaussians' -> '40k'."""
    count = int(setting.split()[0])
    return f"{count/1000:g}k" if count >= 1000 else str(count)


def plot(group, x_key, title, subtitle, path):
    labels = ["Image-GS Gaussian" if entry["method"] == "Image-GS" else entry["method"] for entry in group]
    labels = [f"{label} ({os.path.relpath(entry['path'], os.path.join(ROOT, 'comparisons'))})"
              if labels.count(label) > 1 else label for label, entry in zip(labels, group)]
    with plt.rc_context({"text.usetex": False, "font.family": "sans-serif", "font.size": 10,
                         "axes.titlesize": 11, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "legend.fontsize": 9, "figure.titlesize": 13}):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor=SURFACE, constrained_layout=True)
        for index, (ax, (key, label)) in enumerate(zip(axes.flat, METRICS)):
            ax.set_facecolor(SURFACE)
            for series, ((color, marker), name, entry) in enumerate(zip(PALETTE, labels, group)):
                points = sorted((float(row[x_key]), float(row[key]), row["setting"]) for row in entry["rows"])
                points = [point for point in points if np.isfinite(point[0]) and np.isfinite(point[1])]
                if len(points) == 0:
                    continue
                x, y, settings = zip(*points)
                ax.plot(x, y, color=color, marker=marker, markersize=6, linewidth=1.5, solid_joinstyle="round",
                        solid_capstyle="round", markeredgecolor=SURFACE, markeredgewidth=1.0, label=name, zorder=3)
                if index == 0:  # Primitive counts on the first panel only, to keep the others readable
                    # Alternate series below-right and above-left of their points, so labels of close points never collide
                    offset, align = ((6, -12), "left") if series % 2 == 0 else ((-6, 7), "right")
                    for x_value, y_value, setting in points:
                        ax.annotate(count_label(setting), (x_value, y_value), xytext=offset, textcoords="offset points",
                                    ha=align, fontsize=8, color=color)
            ax.set_xscale("log")
            # Ticks at 1, 2 and 5 per decade with plain numbers: the runs span only one or two decades
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_title(label, color=TEXT_PRIMARY, loc="left")
            ax.set_xlabel("Bits per texel (log scale)", color=TEXT_SECONDARY)
            ax.grid(True, which="major", color=GRID, linewidth=0.8, linestyle="-")
            ax.tick_params(colors=TEXT_SECONDARY, which="both")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
        handles, names = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, names, loc="outside lower center", ncol=len(names), frameon=False, labelcolor=TEXT_PRIMARY)
        fig.suptitle(title, color=TEXT_PRIMARY, x=0.01, ha="left")
        fig.text(0.01, 0.955, subtitle, color=TEXT_SECONDARY, ha="left", va="top", linespacing=1.5)
        fig.get_layout_engine().set(rect=(0, 0, 1, 0.905))
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        plt.close(fig)


def main(args):
    folders = args.comparisons or sorted(glob.glob(os.path.join(ROOT, "comparisons", "*", "*")))
    groups = defaultdict(list)
    for folder in folders:
        entry = read_comparison(folder)
        if entry is not None:
            groups[entry["key"]].append(entry)
    x_key, size_text = SIZES[args.size]
    os.makedirs(args.out_dir, exist_ok=True)
    for texture_dir, group in sorted(groups.items()):
        texture, image = os.path.basename(os.path.dirname(texture_dir)), os.path.basename(texture_dir)
        if len(group) < 2:
            print(f"Skipping {texture} / {image}: only one comparison ('{group[0]['path']}')")
            continue
        if any(x_key not in entry["rows"][0] for entry in group):
            print(f"Skipping {texture} / {image}: no '{x_key}' column, redo the sizes with plot_compression.py --replot")
            continue
        check_scoring(group)
        # Plain Gaussians first, so they keep the Image-GS blue of rd_curves.png
        group.sort(key=lambda entry: (entry["method"] != "Image-GS", entry["method"], entry["path"]))
        suffix = "" if args.size == "file" else f"_{args.size}"
        path = os.path.join(args.out_dir, f"{texture}_{image}{suffix}.png")
        plot(group, x_key, f"{texture} / {image}: Image-GS variants",
             f"{size_text}. Same texture, mask and metrics as rd_curves.png; "
             f"labels on the PSNR panel are primitive counts.\nComparisons: "
             + ", ".join(os.path.relpath(entry["path"], os.path.join(ROOT, "comparisons")) for entry in group), path)
        print(f"Saved '{path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the Image-GS runs of comparisons of the same texture together")
    parser.add_argument("comparisons", nargs="*",
                        help="Comparison folders with a metrics.csv (default: every folder in comparisons/)")
    parser.add_argument("--out_dir", type=str, default=os.path.join(ROOT, "comparisons", "variants"))
    parser.add_argument("--size", type=str, default="file", choices=list(SIZES),
                        help="x-axis: the file as stored (default), on disk with Zstd, or in GPU memory")
    main(parser.parse_args())

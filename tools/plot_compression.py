"""
Score Image-GS (.gtc files) and GPU block formats (tools/baseline_codecs.py output) against the original texture,
on the same empty-area mask that main.py uses, then plot quality against size.
Every method is decoded at the original resolution; lower-resolution baselines are upscaled bilinearly,
the way a GPU magnifies a smaller texture.
"""
import argparse
import csv
import glob
import os
import sys

import cv2
import numpy as np
import torch

IMAGE_GS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "image-gs")
sys.path.insert(0, IMAGE_GS_DIR)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from decode import decode_gtc  # noqa: E402
from lpips import LPIPS  # noqa: E402
from utils.flip import LDRFLIPLoss  # noqa: E402
from utils.gtc_utils import load_gtc  # noqa: E402
from utils.image_utils import get_psnr, get_ssim_map, load_images  # noqa: E402
from utils.mask_utils import compute_valid_masks, masked_mean  # noqa: E402

# Series in a fixed order, colors from the categorical palette, marker shapes so color is never the only cue
SERIES = {
    "Image-GS": ("#2a78d6", "o"),
    "BC1": ("#eb6834", "s"),
    "BC7": ("#1baf7a", "D"),
    "ASTC 4x4": ("#eda100", "^"),
    "ASTC 6x6": ("#e87ba4", "v"),
    "ASTC 8x8": ("#008300", "<"),
    "ASTC 12x12": ("#6250d6", ">"),
}
CODEC_LABELS = {"bc1": "BC1", "bc7": "BC7", "astc4x4": "ASTC 4x4", "astc6x6": "ASTC 6x6", "astc8x8": "ASTC 8x8",
                "astc12x12": "ASTC 12x12"}
METRICS = [("psnr", "PSNR (dB), higher is better"), ("ssim", "SSIM, higher is better"),
           ("lpips", "LPIPS, lower is better"), ("flip", "FLIP, lower is better")]
SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
REFERENCE = "#8c8b86"  # Neutral gray for the uncompressed and original-file reference rows


def score(images, gt_images, mask, lpips_metric, flip_metric):
    """Same metrics as GaussianSplatting2D._evaluate() and _evaluate_extra(), on [3, H, W] images in [0, 1]."""
    if mask is not None:
        images = torch.where(mask, images, gt_images)
    pixel_mask = None if mask is None else mask.any(dim=0)
    result = {"psnr": get_psnr(images, gt_images, mask=mask).item(),
              "ssim": masked_mean(get_ssim_map(images, gt_images, train=False), mask).item()}
    try:
        result["lpips"] = masked_mean(lpips_metric(images[None], gt_images[None]), pixel_mask).item()
        result["flip"] = flip_metric(images[None], gt_images[None], mask=pixel_mask).item()
    except torch.cuda.OutOfMemoryError:
        print("Warning: not enough GPU memory for LPIPS and FLIP at this resolution, they are left empty")
        result["lpips"], result["flip"] = float("nan"), float("nan")
        torch.cuda.empty_cache()
    return result


def plot(rows, title, subtitle, path):
    with plt.rc_context({"text.usetex": False, "font.family": "sans-serif", "font.size": 10,
                         "axes.titlesize": 11, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "legend.fontsize": 9, "figure.titlesize": 13}):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor=SURFACE, constrained_layout=True)
        for ax, (key, label) in zip(axes.flat, METRICS):
            ax.set_facecolor(SURFACE)
            for name, (color, marker) in SERIES.items():
                points = sorted((row["bpp"], row[key]) for row in rows if row["method"] == name and np.isfinite(row[key]))
                if len(points) == 0:
                    continue
                x, y = zip(*points)
                ax.plot(x, y, color=color, marker=marker, markersize=6, linewidth=1.5, solid_joinstyle="round",
                        solid_capstyle="round", markeredgecolor=SURFACE, markeredgewidth=1.0, label=name, zorder=3)
            ax.set_xscale("log")
            ax.set_title(label, color=TEXT_PRIMARY, loc="left")
            ax.set_xlabel("Bits per texel (log scale)", color=TEXT_SECONDARY)
            ax.grid(True, which="major", color=GRID, linewidth=0.8, linestyle="-")
            ax.tick_params(colors=TEXT_SECONDARY, which="both")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False, labelcolor=TEXT_PRIMARY)
        fig.suptitle(title, color=TEXT_PRIMARY, x=0.01, ha="left")
        fig.text(0.01, 0.955, subtitle, color=TEXT_SECONDARY, ha="left", va="top")
        fig.get_layout_engine().set(rect=(0, 0, 1, 0.94))
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        plt.close(fig)


def format_bytes(num_bytes):
    for unit, size in (("MB", 1e6), ("KB", 1e3)):
        if num_bytes >= size:
            return f"{num_bytes/size:.3g} {unit}" if num_bytes < 100*size else f"{num_bytes/size:.0f} {unit}"
    return f"{num_bytes:d} B"


def plot_sizes(rows, image_path, width, height, title, subtitle, path):
    """Dot plot of the file size of every result stored at the original resolution, next to two references."""
    entries = [(row["method"] if row["method"] != "Image-GS" else f"Image-GS, {row['setting']}", row["bytes"],
                *SERIES[row["method"]]) for row in rows if (row["width"], row["height"]) == (width, height)]
    extension = os.path.splitext(image_path)[1].lstrip(".").upper()
    entries += [("Uncompressed RGB (reference)", width * height * 3, REFERENCE, "o"),
                (f"Original {extension} file, lossless (reference)", os.path.getsize(image_path), REFERENCE, "o")]
    entries.sort(key=lambda entry: entry[1])
    with plt.rc_context({"text.usetex": False, "font.family": "sans-serif", "font.size": 10, "axes.titlesize": 11,
                         "xtick.labelsize": 9, "ytick.labelsize": 10, "figure.titlesize": 13}):
        fig_h = 0.42*len(entries) + 1.8
        fig, ax = plt.subplots(figsize=(11, fig_h), facecolor=SURFACE, constrained_layout=True)
        ax.set_facecolor(SURFACE)
        for index, (label, num_bytes, color, marker) in enumerate(entries):
            ax.plot(num_bytes, index, marker=marker, markersize=8, color=color, markeredgecolor=SURFACE,
                    markeredgewidth=1.0, linestyle="none", zorder=3)
            ax.annotate(format_bytes(num_bytes), (num_bytes, index), xytext=(9, 0), textcoords="offset points",
                        va="center", ha="left", color=TEXT_PRIMARY, fontsize=9)
        ax.set_yticks(range(len(entries)), [entry[0] for entry in entries], color=TEXT_PRIMARY)
        ax.set_xscale("log")
        sizes = [entry[1] for entry in entries]
        ax.set_xlim(min(sizes) / 2.0, max(sizes) * 4.0)  # Room for the value labels
        ticks = [10.0**exponent for exponent in range(0, 12) if min(sizes) / 2.0 <= 10.0**exponent <= max(sizes) * 4.0]
        ax.set_xticks(ticks, [format_bytes(int(tick)) for tick in ticks])
        ax.minorticks_off()
        ax.set_xlabel("Size on disk (log scale)", color=TEXT_SECONDARY)
        ax.set_ylim(-0.6, len(entries) - 0.4)
        ax.grid(True, axis="both", color=GRID, linewidth=0.8, linestyle="-")
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", colors=TEXT_SECONDARY)
        ax.tick_params(axis="y", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        # Title and subtitle sit in a fixed band above the plot, aligned with the figure's left edge
        fig.text(0.01, 1.0 - 0.2/fig_h, title, color=TEXT_PRIMARY, ha="left", va="top", fontsize=13)
        fig.text(0.01, 1.0 - 0.5/fig_h, subtitle, color=TEXT_SECONDARY, ha="left", va="top", fontsize=10)
        fig.get_layout_engine().set(rect=(0, 0, 1, 1.0 - 0.85/fig_h))
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        plt.close(fig)


def main(args):
    device = args.device
    gt, input_channels, image_fnames, _ = load_images(args.image)
    if sum(input_channels) != 3:
        raise ValueError(f"Expected an RGB texture but '{args.image}' has {sum(input_channels)} channels")
    height, width = gt.shape[1:]
    mask = None
    if args.empty_mode != "none":
        tol = args.empty_detail_tol if args.empty_mode == "blurred" else args.empty_tol
        mask = compute_valid_masks(gt, input_channels, image_fnames, args.image, args.empty_mode, tol, args.empty_dilate)
        mask = torch.from_numpy(mask).to(device)
        print(f"Scoring on used texels only ('{args.empty_mode}' mask, {100.0*mask[0].float().mean().item():.2f}% of texels)")
    gt_images = torch.from_numpy(gt).to(device)
    lpips_metric = LPIPS(net="alex", spatial=True, verbose=False).to(device).eval()
    flip_metric = LDRFLIPLoss().to(device).eval()
    rows = []
    with torch.no_grad():
        gtc_files = sorted(glob.glob(os.path.join(args.gaussians, "*", "gaussians.gtc"))) if args.gaussians else []
        for gtc_file in gtc_files:
            gaussians = load_gtc(gtc_file)
            images, _ = decode_gtc(gaussians, render_height=height, device=device)
            images = torch.pow(torch.clamp(images, 0.0, 1.0), 1.0/gaussians.gamma)
            rows.append({"method": "Image-GS", "setting": f"{len(gaussians.xy):d} Gaussians", "width": width,
                         "height": height, "bytes": os.path.getsize(gtc_file), "file": gtc_file,
                         **score(images, gt_images, mask, lpips_metric, flip_metric)})
        manifest = os.path.join(args.baselines, "baselines.csv")
        with open(manifest, "r", encoding="utf-8") as file:
            baselines = list(csv.DictReader(file))
        for baseline in baselines:
            decoded = cv2.imread(os.path.join(args.baselines, baseline["decoded_file"]))[..., ::-1]
            if decoded.shape[:2] != (height, width):
                decoded = cv2.resize(decoded, (width, height), interpolation=cv2.INTER_LINEAR)
            images = torch.from_numpy(np.ascontiguousarray(decoded).transpose(2, 0, 1)).to(device=device, dtype=torch.float32) / 255.0
            rows.append({"method": CODEC_LABELS[baseline["codec"]], "setting": f"{baseline['width']}x{baseline['height']}",
                         "width": int(baseline["width"]), "height": int(baseline["height"]), "bytes": int(baseline["bytes"]),
                         "file": os.path.join(args.baselines, baseline["compressed_file"]),
                         **score(images, gt_images, mask, lpips_metric, flip_metric)})
            torch.cuda.empty_cache()
    for row in rows:
        row["bpp"] = 8.0 * row["bytes"] / (height * width)  # Relative to the original texel count for every method
    rows.sort(key=lambda row: (list(SERIES).index(row["method"]), row["bpp"]))
    os.makedirs(args.out_dir, exist_ok=True)
    fields = ["method", "setting", "width", "height", "bytes", "bpp", "psnr", "ssim", "lpips", "flip", "file"]
    with open(os.path.join(args.out_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fields} for row in rows])
    for row in rows:
        print(f"{row['method']:11s} {row['setting']:16s} {row['bytes']/1e3:10.2f} KB {row['bpp']:7.3f} bpp | "
              f"PSNR {row['psnr']:6.2f} | SSIM {row['ssim']:.4f} | LPIPS {row['lpips']:.4f} | FLIP {row['flip']:.4f}")
    texture = os.path.basename(os.path.dirname(os.path.abspath(args.image)))
    title = f"{texture} / {image_fnames[0]}: quality against size"
    scored_on = "all texels" if mask is None else f"used texels only ('{args.empty_mode}' mask)"
    subtitle = (f"{width}x{height} texture, scored on {scored_on}. BC/ASTC points are the texture compressed at "
                f"1/1 to 1/{max(width // min(row['width'] for row in rows), 1)} resolution and upscaled bilinearly.")
    plot(rows, title, subtitle, os.path.join(args.out_dir, "rd_curves.png"))
    plot_sizes(rows, args.image, width, height, f"{texture} / {image_fnames[0]}: size of each {width}x{height} result",
               "File size on disk, headers included. Image-GS files decode to this resolution. "
               "Quality of each result: rd_curves.png", os.path.join(args.out_dir, "sizes.png"))
    print(f"Saved 'metrics.csv', 'rd_curves.png' and 'sizes.png' in '{args.out_dir}'")


if __name__ == "__main__":
    torch.hub.set_dir(os.path.join(IMAGE_GS_DIR, "models", "torch"))  # Reuse the LPIPS backbone that main.py downloaded
    parser = argparse.ArgumentParser(description="Score and plot Image-GS against BC1, BC7 and ASTC")
    parser.add_argument("image", type=str, help="Original texture image file")
    parser.add_argument("--baselines", type=str, required=True, help="Output folder of tools/baseline_codecs.py")
    parser.add_argument("--gaussians", type=str, default=None,
                        help="Experiment folder of main.py (results/<exp_name>), every run inside with a gaussians.gtc is scored")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--empty_mode", type=str, default="none", help="Same as main.py --empty_mode")
    parser.add_argument("--empty_tol", type=float, default=2.0, help="Same as main.py --empty_tol")
    parser.add_argument("--empty_detail_tol", type=float, default=0.25, help="Same as main.py --empty_detail_tol")
    parser.add_argument("--empty_dilate", type=int, default=2, help="Same as main.py --empty_dilate")
    parser.add_argument("--device", type=str, default="cuda:0")
    main(parser.parse_args())

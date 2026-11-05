"""
Score Image-GS (.gtc files) and GPU block formats (tools/baseline_codecs.py output) against the original texture,
on the same empty-area mask that main.py uses, then plot quality against the size of the compressed file. The size
of every full-resolution result is also plotted two more ways:
  on disk         every file compressed losslessly with Zstd, the supercompression step of a shipping pipeline
  in GPU memory   BC/ASTC as the hardware samples them; Image-GS as its Gaussians plus the per-tile Gaussian lists a
                  shader needs to sample them directly
Every method is decoded at the original resolution; lower-resolution baselines are upscaled bilinearly,
the way a GPU magnifies a smaller texture.
"""
import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np
import torch

IMAGE_GS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "image-gs")
sys.path.insert(0, IMAGE_GS_DIR)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from decode import decode_gtc  # noqa: E402
from lpips import LPIPS  # noqa: E402
from model import get_pixel_scale  # noqa: E402
from utils.flip import LDRFLIPLoss  # noqa: E402
from utils.gtc_utils import load_gtc  # noqa: E402
from utils.image_utils import get_psnr, get_ssim_map, load_images  # noqa: E402
from utils.mask_utils import compute_valid_masks, masked_mean  # noqa: E402

# Series in a fixed order, colors from the categorical palette, marker shapes so color is never the only cue
SERIES = {
    "Image-GS Beta": ("#2a78d6", "o"),
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
FIELDS = ["method", "setting", "width", "height", "bytes", "bpp", "zstd_bytes", "zstd_bpp", "gpu_bytes", "gpu_bpp",
          "table_bytes", "gaussians_per_tile", "psnr", "ssim", "lpips", "flip", "file"]
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


def score_all(args, gt, input_channels, image_fnames):
    """Decode every Image-GS run and baseline at the texture's resolution and score it."""
    device = args.device
    height, width = gt.shape[1:]
    mask = None
    if args.empty_mode != "none":
        tol = args.empty_detail_tol if args.empty_mode == "blurred" else args.empty_tol
        mask = compute_valid_masks(gt, input_channels, image_fnames, args.image, args.empty_mode, tol, args.empty_dilate,
                                   args.empty_uv_file)
        mask = torch.from_numpy(mask).to(device)
        print(f"Scoring on used texels only ('{args.empty_mode}' mask, {100.0*mask[0].float().mean().item():.2f}% of texels)")
    gt_images = torch.from_numpy(gt).to(device)
    lpips_metric = LPIPS(net="alex", spatial=True, verbose=False).to(device).eval()
    flip_metric = LDRFLIPLoss().to(device).eval()
    rows = []
    with torch.no_grad():
        gtc_files = sorted(glob.glob(os.path.join(args.gaussians, "*", "gaussians.gtc"))) if args.gaussians else []
        for gtc_file in gtc_files:
            try:
                gaussians = load_gtc(gtc_file)
            except ValueError as error:  # e.g. a plain Gaussian run from the main branch
                print(f"Skipping {error}")
                continue
            images, _ = decode_gtc(gaussians, render_height=height, device=device)
            images = torch.pow(torch.clamp(images, 0.0, 1.0), 1.0/gaussians.gamma)
            rows.append({"method": "Image-GS Beta", "setting": f"{len(gaussians.xy):d} Gaussians", "width": width,
                         "height": height, "file": gtc_file, **score(images, gt_images, mask, lpips_metric, flip_metric)})
        manifest = os.path.join(args.baselines, "baselines.csv")
        with open(manifest, "r", encoding="utf-8") as file:
            baselines = list(csv.DictReader(file))
        for baseline in baselines:
            decoded = cv2.imread(os.path.join(args.baselines, baseline["decoded_file"]))[..., ::-1]
            if decoded.shape[:2] != (height, width):
                decoded = cv2.resize(decoded, (width, height), interpolation=cv2.INTER_LINEAR)
            images = torch.from_numpy(np.ascontiguousarray(decoded).transpose(2, 0, 1)).to(device=device, dtype=torch.float32) / 255.0
            rows.append({"method": CODEC_LABELS[baseline["codec"]], "setting": f"{baseline['width']}x{baseline['height']}",
                         "width": int(baseline["width"]), "height": int(baseline["height"]),
                         "file": os.path.join(args.baselines, baseline["compressed_file"]),
                         **score(images, gt_images, mask, lpips_metric, flip_metric)})
            torch.cuda.empty_cache()
    return rows


def read_rows(path):
    """Quality metrics of an earlier run, so that sizes and plots can be redone without the GPU."""
    with open(path, "r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["width"], row["height"] = int(row["width"]), int(row["height"])
        if row["method"] == "Image-GS":  # Written before the Beta label
            row["method"] = "Image-GS Beta"
        for key, _ in METRICS:
            row[key] = float(row[key])
    return rows


def zstd_size(path, level):
    """Size of the file after lossless Zstd compression."""
    command = ["zstd", f"-{level:d}", "-T0", "-q", "-c", path] + (["--ultra"] if level > 19 else [])
    return len(subprocess.run(command, capture_output=True, check=True).stdout)


def tile_table_size(gaussians, height, width, tile):
    """
    Bytes of the per-tile Gaussian lists that a shader needs to sample the Gaussians directly at height x width:
    one Gaussian id per (Gaussian, tile) pair, 16-bit up to 65536 Gaussians and 32-bit above, plus one 32-bit list
    offset per tile. The tiles a Gaussian covers follow the rasterizer, whose Beta kernel ends inside this box: a 3-sigma radius from the eigenvalues of its
    covariance (compute_cov2d_bounds() in gsplat/cuda/csrc/helpers.cuh) and the same bounding box (get_bbox()).
    Returns the size and the average number of Gaussians per tile, which a sample has to evaluate.
    """
    scale = get_pixel_scale(gaussians.scale.astype(np.float32), inverse_scale=gaussians.inverse_scale,
                            upsample_ratio=float(height)/gaussians.img_h)
    var = scale.astype(np.float32)**2  # The covariance R S S R^T has the eigenvalues sx^2 and sy^2
    mid, det = 0.5*(var[:, 0] + var[:, 1]), var[:, 0]*var[:, 1]
    radius = np.ceil(3.0*np.sqrt(mid + np.sqrt(np.maximum(0.1, mid*mid - det))))
    tiles_x, tiles_y = (width + tile - 1) // tile, (height + tile - 1) // tile
    center_x, center_y = gaussians.xy[:, 0]*width/tile, gaussians.xy[:, 1]*height/tile
    radius = radius/tile
    span_x = np.clip(np.trunc(center_x + radius + 1), 0, tiles_x) - np.clip(np.trunc(center_x - radius), 0, tiles_x)
    span_y = np.clip(np.trunc(center_y + radius + 1), 0, tiles_y) - np.clip(np.trunc(center_y - radius), 0, tiles_y)
    num_pairs = int((np.maximum(span_x, 0)*np.maximum(span_y, 0))[det > 0].sum())
    id_bytes = 2 if len(gaussians.xy) <= 2**16 else 4
    return num_pairs*id_bytes + tiles_x*tiles_y*4, num_pairs / (tiles_x*tiles_y)


def add_sizes(rows, height, width, zstd_level, table_tile):
    """Size of every result as stored, on disk (Zstd) and in GPU memory. Unavailable sizes are left as NaN."""
    print(f"Compressing every result with Zstd -{zstd_level:d}")
    for row in rows:
        row.update(bytes=float("nan"), zstd_bytes=float("nan"), gpu_bytes=float("nan"), table_bytes="",
                   gaussians_per_tile="")
        if not os.path.isfile(row["file"]):
            print(f"Warning: no size for {row['method']}, {row['setting']}: '{row['file']}' no longer exists")
        else:
            row["bytes"] = os.path.getsize(row["file"])
            row["zstd_bytes"] = zstd_size(row["file"], zstd_level)
            row["gpu_bytes"] = row["bytes"]
            if row["method"] == "Image-GS Beta":
                try:
                    gaussians = load_gtc(row["file"])
                    row["table_bytes"], row["gaussians_per_tile"] = tile_table_size(gaussians, height, width, table_tile)
                    row["gpu_bytes"] += row["table_bytes"]
                except ValueError as error:
                    print(f"Warning: no GPU memory size, {error}")
                    row["gpu_bytes"] = float("nan")
        for key in ("bytes", "zstd_bytes", "gpu_bytes"):  # Relative to the original texel count for every method
            row[key.replace("bytes", "bpp")] = 8.0 * row[key] / (height * width)
    return rows


def plot(rows, x_key, title, subtitle, path):
    with plt.rc_context({"text.usetex": False, "font.family": "sans-serif", "font.size": 10,
                         "axes.titlesize": 11, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "legend.fontsize": 9, "figure.titlesize": 13}):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor=SURFACE, constrained_layout=True)
        for ax, (key, label) in zip(axes.flat, METRICS):
            ax.set_facecolor(SURFACE)
            for name, (color, marker) in SERIES.items():
                points = sorted((row[x_key], row[key]) for row in rows
                                if row["method"] == name and np.isfinite(row[x_key]) and np.isfinite(row[key]))
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
        fig.text(0.01, 0.955, subtitle, color=TEXT_SECONDARY, ha="left", va="top", linespacing=1.5)
        fig.get_layout_engine().set(rect=(0, 0, 1, 0.905))
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        plt.close(fig)


def format_bytes(num_bytes):
    for unit, size in (("MB", 1e6), ("KB", 1e3)):
        if num_bytes >= size:
            return f"{num_bytes/size:.3g} {unit}" if num_bytes < 100*size else f"{num_bytes/size:.0f} {unit}"
    return f"{int(num_bytes):d} B"


def plot_sizes(rows, image_path, width, height, title, subtitle, path):
    """
    Dot plot of every result stored at the original resolution: its size on disk (filled) and in GPU memory (hollow),
    next to two references.
    """
    entries = [(row["method"] if row["method"] != "Image-GS Beta" else f"Image-GS Beta, {row['setting']}", row["zstd_bytes"],
                row["gpu_bytes"], *SERIES[row["method"]]) for row in rows
               if (row["width"], row["height"]) == (width, height) and np.isfinite(row["zstd_bytes"])]
    extension = os.path.splitext(image_path)[1].lstrip(".").upper()
    entries += [(f"Original {extension} file, lossless (reference)", os.path.getsize(image_path), float("nan"), REFERENCE, "o"),
                ("Uncompressed RGBA8 (reference)", float("nan"), width * height * 4, REFERENCE, "o")]
    entries.sort(key=lambda entry: entry[1] if np.isfinite(entry[1]) else entry[2])
    with plt.rc_context({"text.usetex": False, "font.family": "sans-serif", "font.size": 10, "axes.titlesize": 11,
                         "xtick.labelsize": 9, "ytick.labelsize": 10, "legend.fontsize": 9, "figure.titlesize": 13}):
        fig_h = 0.42*len(entries) + 1.8
        fig, ax = plt.subplots(figsize=(11, fig_h), facecolor=SURFACE, constrained_layout=True)
        ax.set_facecolor(SURFACE)
        for index, (label, disk, gpu, color, marker) in enumerate(entries):
            if np.isfinite(disk) and np.isfinite(gpu):
                ax.plot([disk, gpu], [index, index], color=color, linewidth=1.2, alpha=0.5, zorder=2)
            if np.isfinite(disk):
                ax.plot(disk, index, marker=marker, markersize=8, color=color, markeredgecolor=SURFACE,
                        markeredgewidth=1.0, linestyle="none", zorder=3)
            if np.isfinite(gpu):
                ax.plot(gpu, index, marker=marker, markersize=8, color=color, markerfacecolor=SURFACE,
                        markeredgecolor=color, markeredgewidth=1.6, linestyle="none", zorder=3)
            values = [f"{name} {format_bytes(value)}" for name, value in (("disk", disk), ("GPU", gpu)) if np.isfinite(value)]
            ax.annotate(" · ".join(values), (np.nanmax([disk, gpu]), index), xytext=(9, 0), textcoords="offset points",
                        va="center", ha="left", color=TEXT_PRIMARY, fontsize=9)
        ax.set_yticks(range(len(entries)), [entry[0] for entry in entries], color=TEXT_PRIMARY)
        ax.set_xscale("log")
        sizes = [size for entry in entries for size in entry[1:3] if np.isfinite(size)]
        ax.set_xlim(min(sizes) / 2.0, max(sizes) * 12.0)  # Room for the value labels
        ticks = [10.0**exponent for exponent in range(0, 12) if min(sizes) / 2.0 <= 10.0**exponent <= max(sizes) * 12.0]
        ax.set_xticks(ticks, [format_bytes(int(tick)) for tick in ticks])
        ax.minorticks_off()
        ax.set_xlabel("Size (log scale)", color=TEXT_SECONDARY)
        ax.set_ylim(-0.6, len(entries) - 0.4)
        ax.grid(True, axis="both", color=GRID, linewidth=0.8, linestyle="-")
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", colors=TEXT_SECONDARY)
        ax.tick_params(axis="y", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        legend = [Line2D([], [], marker="o", markersize=8, color=TEXT_SECONDARY, markeredgecolor=SURFACE,
                         linestyle="none", label="On disk, Zstd"),
                  Line2D([], [], marker="o", markersize=8, color=TEXT_SECONDARY, markerfacecolor=SURFACE,
                         markeredgewidth=1.6, linestyle="none", label="In GPU memory")]
        ax.legend(handles=legend, loc="lower right", frameon=False, labelcolor=TEXT_PRIMARY)
        # Title and subtitle sit in a fixed band above the plot, aligned with the figure's left edge
        fig.text(0.01, 1.0 - 0.2/fig_h, title, color=TEXT_PRIMARY, ha="left", va="top", fontsize=13)
        fig.text(0.01, 1.0 - 0.5/fig_h, subtitle, color=TEXT_SECONDARY, ha="left", va="top", fontsize=10)
        fig.get_layout_engine().set(rect=(0, 0, 1, 1.0 - 0.85/fig_h))
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        plt.close(fig)


def main(args):
    if shutil.which("zstd") is None:
        raise RuntimeError("The zstd command is missing, install it with: sudo dnf install zstd")
    gt, input_channels, image_fnames, _ = load_images(args.image)
    if sum(input_channels) != 3:
        raise ValueError(f"Expected an RGB texture but '{args.image}' has {sum(input_channels)} channels")
    height, width = gt.shape[1:]
    metrics_path = os.path.join(args.out_dir, "metrics.csv")
    rows = read_rows(metrics_path) if args.replot else score_all(args, gt, input_channels, image_fnames)
    rows = add_sizes(rows, height, width, args.zstd_level, args.table_tile)
    rows.sort(key=lambda row: (list(SERIES).index(row["method"]), row["bpp"]))
    os.makedirs(args.out_dir, exist_ok=True)
    with open(metrics_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in FIELDS} for row in rows])
    for row in rows:
        print(f"{row['method']:13s} {row['setting']:16s} file {row['bytes']/1e3:9.1f} KB | Zstd {row['zstd_bytes']/1e3:9.1f} KB"
              f" | GPU {row['gpu_bytes']/1e3:9.1f} KB | PSNR {row['psnr']:6.2f} | SSIM {row['ssim']:.4f} | "
              f"LPIPS {row['lpips']:.4f} | FLIP {row['flip']:.4f}")
    texture = os.path.basename(os.path.dirname(os.path.abspath(args.image)))
    name = f"{texture} / {image_fnames[0]}"
    scored_on = "all texels" if args.empty_mode == "none" else f"used texels only ('{args.empty_mode}' mask)"
    scales = max(width // min(row["width"] for row in rows), 1)
    common = (f"{width}x{height} texture, scored on {scored_on}. BC/ASTC points are the texture compressed at "
              f"1/1 to 1/{scales} resolution and upscaled bilinearly.")
    plot(rows, "bpp", f"{name}: quality against size",
         f"Size: the compressed file as stored (.gtc, .dds, .astc, headers included). Its size on disk with Zstd and in "
         f"GPU memory: sizes.png\n{common}", os.path.join(args.out_dir, "rd_curves.png"))
    plot_sizes(rows, args.image, width, height, f"{name}: size of each {width}x{height} result",
               f"Filled: on disk, compressed with Zstd -{args.zstd_level:d}. Hollow: in GPU memory, Image-GS with its "
               f"{args.table_tile}x{args.table_tile} px tile lists. Headers included.",
               os.path.join(args.out_dir, "sizes.png"))
    print(f"Saved 'metrics.csv', 'rd_curves.png' and 'sizes.png' in '{args.out_dir}'")


if __name__ == "__main__":
    torch.hub.set_dir(os.path.join(IMAGE_GS_DIR, "models", "torch"))  # Reuse the LPIPS backbone that main.py downloaded
    parser = argparse.ArgumentParser(description="Score and plot Image-GS against BC1, BC7 and ASTC")
    parser.add_argument("image", type=str, help="Original texture image file")
    parser.add_argument("--baselines", type=str, default=None, help="Output folder of tools/baseline_codecs.py")
    parser.add_argument("--gaussians", type=str, default=None,
                        help="Experiment folder of main.py (results/<exp_name>), every run inside with a gaussians.gtc is scored")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--replot", action="store_true",
                        help="Keep the scores in <out_dir>/metrics.csv and only redo the sizes and plots, without the GPU. "
                             "Pass the same --empty_mode, it is named in the plot subtitles")
    parser.add_argument("--zstd_level", type=int, default=19, help="Zstd level for the size on disk, 1 to 22")
    parser.add_argument("--table_tile", type=int, default=16,
                        help="Tile size in pixels of the Gaussian lists counted in GPU memory (default: the rasterizer's 16)")
    parser.add_argument("--empty_mode", type=str, default="none", help="Same as main.py --empty_mode")
    parser.add_argument("--empty_tol", type=float, default=2.0, help="Same as main.py --empty_tol")
    parser.add_argument("--empty_detail_tol", type=float, default=0.25, help="Same as main.py --empty_detail_tol")
    parser.add_argument("--empty_uv_file", type=str, default="", help="Same as main.py --empty_uv_file")
    parser.add_argument("--empty_dilate", type=int, default=2, help="Same as main.py --empty_dilate")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    if not args.replot and args.baselines is None:
        parser.error("--baselines is required unless --replot is given")
    main(args)

"""
Render a Blender scene with four textures and compose two comparison figures:
  original      the texture every method was scored against
  BC/ASTC       the best result (by --by) whose file is no larger than the Image-GS files; with --resolution full
                only results at the texture's own resolution count, and the smallest one is used if none is that small
  Image-GS      one run of each comparison given, e.g. Gaussian and Beta, at the same primitive count
<out_dir>/<name>_render.png shows the four renders, cropped to the object, <name>_zoom.png the same with a detail of
each rendered at --zoom_percent resolution, picture in picture. Renders are kept in <out_dir>/<name>/.
Image-GS textures are the PNGs that tools/decode.py writes next to each gaussians.gtc: decode them first, each on
the branch whose kernel wrote it, e.g.  python tools/decode.py <gaussians.gtc> --render_height 4096 --format png
The .blend file is only read; its camera, Workbench or other render settings are used as they are.
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.patches import ConnectionPatch, Rectangle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLENDER_SCRIPT = os.path.join(ROOT, "tools", "blender_render.py")
METRICS = {"psnr": ("PSNR", "dB", max), "ssim": ("SSIM", "", max), "lpips": ("LPIPS", "", min), "flip": ("FLIP", "", min)}
SURFACE, TEXT_PRIMARY, TEXT_SECONDARY = "#fcfcfb", "#0b0b0b", "#52514e"
ACCENT = "#e8175d"  # Zoom frame and arrow, with a white halo so they stay visible on any texture
HALO = [patheffects.withStroke(linewidth=4.5, foreground="white")]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr")


def read_rows(folder):
    with open(os.path.join(folder, "metrics.csv"), "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def format_bytes(num_bytes):
    for unit, size in (("MB", 1e6), ("KB", 1e3)):
        if num_bytes >= size:
            return f"{num_bytes/size:.3g} {unit}"
    return f"{int(num_bytes):d} B"


def stats(row, metric):
    label, unit, _ = METRICS[metric]
    shown = [("psnr", "PSNR {:.2f} dB"), ("lpips", "LPIPS {:.3f}")]
    if metric not in ("psnr", "lpips"):
        shown.append((metric, f"{label} {{:.3f}}{unit}"))
    return " · ".join([format_bytes(float(row["bytes"]))] + [text.format(float(row[key])) for key, text in shown])


def find_texture(texture_dir):
    """Original texture of non_gaussian_compressions/<texture>/<image>, or its capped copy for <image>_max<cap>."""
    texture, name = os.path.basename(os.path.dirname(texture_dir)), os.path.basename(texture_dir)
    capped = re.fullmatch(r"(.+)_max(\d+)", name)
    pattern = (os.path.join(ROOT, "capped_textures", capped.group(2), texture, capped.group(1) + ".*") if capped
               else os.path.join(ROOT, texture, name + ".*"))
    found = [path for path in sorted(glob.glob(pattern)) if path.lower().endswith(IMAGE_EXTENSIONS)]
    if not found:
        raise FileNotFoundError(f"No texture matches '{pattern}', pass it with --texture")
    return found[0]


def select(args):
    """The four textures, each with its label, details and file."""
    comparisons = [(folder, read_rows(folder)) for folder in args.comparisons]
    runs = [[row for row in rows if row["method"].startswith("Image-GS")] for _, rows in comparisons]
    counts = set.intersection(*[{int(row["setting"].split()[0]) for row in rows} for rows in runs])
    if not counts:
        raise ValueError("The comparisons have no primitive count in common")
    count = args.count if args.count is not None else max(counts)
    if count not in counts:
        raise ValueError(f"Not every comparison has a {count} run, the common counts are {sorted(counts)}")
    chosen = [next(row for row in rows if int(row["setting"].split()[0]) == count) for rows in runs]
    # Plain Gaussians first, like compare_variants.py
    order = sorted(range(len(chosen)), key=lambda index: (chosen[index]["method"] != "Image-GS", chosen[index]["method"]))
    chosen = [chosen[index] for index in order]
    baselines = [row for row in comparisons[0][1] if not row["method"].startswith("Image-GS")]
    texture_dir = os.path.dirname(os.path.abspath(baselines[0]["file"]))
    original = args.texture or find_texture(texture_dir)
    stem = os.path.splitext(os.path.basename(original))[0]
    height, width = cv2.imread(original, cv2.IMREAD_UNCHANGED).shape[:2]
    if args.resolution == "full":  # Only BC/ASTC at the resolution the Image-GS runs are decoded at
        baselines = [row for row in baselines if (int(row["width"]), int(row["height"])) == (width, height)]
    budget = max(float(row["bytes"]) for row in chosen)
    candidates = [row for row in baselines if float(row["bytes"]) <= budget]
    metric = METRICS[args.by][0]
    among = f"the {width}×{height} results" if args.resolution == "full" else "the results"
    if candidates:
        best = METRICS[args.by][2](candidates, key=lambda row: float(row[args.by]))
        note = f"BC/ASTC: the best {metric} among {among} no larger than the Image-GS files ({format_bytes(budget)})."
    elif args.resolution == "full":
        best = min(baselines, key=lambda row: float(row["bytes"]))  # The closest one to the Image-GS files
        note = (f"BC/ASTC: the smallest {width}×{height} result, {float(best['bytes']) / budget:.2f}× the largest "
                f"Image-GS file ({format_bytes(budget)}), since none at this resolution is as small.")
    else:
        raise ValueError(f"No BC/ASTC result is as small as the Image-GS files ({format_bytes(budget)})")
    entries = [{"key": "original", "title": "Original texture", "details": f"{width}×{height}, the reference",
                "texture": original},
               {"key": os.path.splitext(os.path.basename(best["file"]))[0],
                "title": f"{best['method']} at {best['setting']}".replace("x", "×"), "details": stats(best, args.by),
                "texture": os.path.splitext(best["file"])[0] + ".png"}]
    for row in chosen:
        name = "Image-GS Gaussian" if row["method"] == "Image-GS" else row["method"]
        entries.append({"key": name.lower().replace(" ", "-"), "title": f"{name}, {count//1000:d}k primitives",
                        "details": stats(row, args.by), "texture": os.path.join(os.path.dirname(row["file"]), stem + ".png")})
    for entry in entries[1:]:
        if not os.path.isfile(entry["texture"]):
            hint = ("decode it with: python tools/decode.py <gaussians.gtc> --render_height "
                    f"{height} --format png, on the branch whose kernel wrote it") if entry["key"].startswith("image-gs") else ""
            raise FileNotFoundError(f"No decoded texture '{entry['texture']}' {hint}")
    texture_name = f"{os.path.basename(os.path.dirname(texture_dir))} / {os.path.basename(texture_dir)}"
    return entries, texture_name, os.path.basename(original), note, (width, height)


def run_blender(args, image_name, jobs, work_dir, tag):
    spec_path = os.path.join(work_dir, f"jobs_{tag}.json")
    with open(spec_path, "w", encoding="utf-8") as file:
        json.dump({"image": image_name, "jobs": jobs}, file, indent=1)
    command = [args.blender, "-b", args.blend, "--python", BLENDER_SCRIPT, "--", spec_path]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or any(not os.path.isfile(job["output"]) for job in jobs):
        raise RuntimeError(f"Blender failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}")


def load_render(path):
    """RGBA render as float RGB composited on the figure background, plus its alpha."""
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    rgb = image[..., 2::-1].astype(np.float32) / 255.0
    alpha = image[..., 3:4].astype(np.float32) / 255.0 if image.shape[2] == 4 else np.ones_like(rgb[..., :1])
    surface = np.array(matplotlib.colors.to_rgb(SURFACE), dtype=np.float32)
    return rgb * alpha + surface * (1.0 - alpha), alpha[..., 0]


def pick_zoom(rgb, alpha, size):
    """
    Square window of size x size pixels with the most texture detail (mean gradient magnitude) that lies on the
    object, away from its silhouette. Returns (x0, y0) in pixels.
    """
    gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    detail = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    covered = (alpha > 0.99).astype(np.float32)
    detail_sum, covered_sum = cv2.integral(detail * covered), cv2.integral(covered)
    height, width = gray.shape
    margin = size // 4
    best, best_score = None, -1.0
    for y0 in range(margin, height - size - margin, 4):
        for x0 in range(margin, width - size - margin, 4):
            y1, x1 = y0 + size, x0 + size
            area = covered_sum[y1, x1] - covered_sum[y0, x1] - covered_sum[y1, x0] + covered_sum[y0, x0]
            if area < 0.98 * size * size:
                continue
            score = detail_sum[y1, x1] - detail_sum[y0, x1] - detail_sum[y1, x0] + detail_sum[y0, x0]
            if score > best_score:
                best, best_score = (x0, y0), score
    if best is None:
        raise ValueError("No zoom window fits inside the object, pass one with --zoom")
    return best


def object_box(renders):
    """Bounding box (x0, y0, x1, y1) of the object over every render."""
    alpha = np.max([alpha for _, alpha in renders], axis=0)
    ys, xs = np.nonzero(alpha > 0.01)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def crop(rgb, region):
    """rgb cut to region (x0, y0, x1, y1), padded with the figure background where region leaves the frame."""
    x0, y0, x1, y1 = region
    height, width = rgb.shape[:2]
    pad = ((max(0, -y0), max(0, y1 - height)), (max(0, -x0), max(0, x1 - width)), (0, 0))
    surface = np.array(matplotlib.colors.to_rgb(SURFACE), dtype=np.float32)
    padded = np.stack([np.pad(rgb[..., c], pad[:2], constant_values=surface[c]) for c in range(3)], axis=-1)
    return padded[y0 + pad[0][0]:y1 + pad[0][0], x0 + pad[1][0]:x1 + pad[1][0]]


def zoom_layout(obj, box, inset_size):
    """
    Inset square next to the object, on the side of the zoomed area, level with it. Returns the inset (x, y, side) and
    the region that holds object and inset, both in frame pixels.
    """
    ox0, oy0, ox1, oy1 = obj
    side = int(round(inset_size * (oy1 - oy0)))
    gap = int(round(0.08 * (oy1 - oy0)))
    box_cx, box_cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    x = ox1 + gap if box_cx >= (ox0 + ox1) / 2 else ox0 - gap - side
    y = int(round(min(max(box_cy - side / 2, oy0), max(oy1 - side, oy0))))
    margin = int(round(0.04 * (oy1 - oy0)))
    region = (min(ox0, x) - margin, min(oy0, y) - margin, max(ox1, x + side) + margin, max(oy1, y + side) + margin)
    return (x, y, side), region


def figure(entries, panels, title, subtitle, path, zoom=None):
    """
    The four panels with their titles, 2x2 for wide panels and in a row for tall ones. zoom: the zoomed area "box",
    the "inset" (x, y, side) and the detail "crops", in panel pixels.
    """
    height, width = panels[0].shape[:2]
    cols = 4 if width < height else 2
    rows = len(panels) // cols
    total_w, band, gap = 14.0, 1.75, 0.75  # Inches: figure width, title band, space for each panel's title
    panel_w = (total_w - 0.3) / cols
    panel_h = panel_w * height / width
    fig = plt.figure(figsize=(total_w, rows * (panel_h + gap) + band - gap + 0.15), facecolor=SURFACE)
    fig_h = fig.get_figheight()
    grid = fig.add_gridspec(rows, cols, left=0.01, right=0.99, bottom=0.1 / fig_h, top=1.0 - band / fig_h,
                            hspace=gap / panel_h, wspace=0.3 / total_w * cols / max(cols - 1, 1))
    for index, (entry, rgb) in enumerate(zip(entries, panels)):
        ax = fig.add_subplot(grid[index // cols, index % cols])
        ax.imshow(rgb, interpolation="lanczos")
        ax.set_axis_off()
        ax.annotate(entry["title"], xy=(0, 1), xycoords="axes fraction", xytext=(0, 20), textcoords="offset points",
                    fontsize=12, fontweight="bold", color=TEXT_PRIMARY, va="bottom")
        ax.annotate(entry["details"], xy=(0, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points",
                    fontsize=10, color=TEXT_SECONDARY, va="bottom")
        if zoom is None:
            continue
        x0, y0, x1, y1 = zoom["box"]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=ACCENT, linewidth=2.0,
                               path_effects=HALO, zorder=4))
        ix, iy, side = zoom["inset"]
        # Axes fractions run upwards from the bottom-left, image pixels downwards from the top-left
        inset = ax.inset_axes([ix / width, 1.0 - (iy + side) / height, side / width, side / height])
        inset.imshow(zoom["crops"][index], interpolation="lanczos")
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_edgecolor(ACCENT)
            spine.set_linewidth(2.5)
        # Arrow from the inset's edge facing the zoomed area to the area's facing edge
        dx = (x0 + x1) / 2 - (ix + side / 2)
        dy = (y0 + y1) / 2 - (iy + side / 2)
        if abs(dx) > abs(dy):
            start, end = (1.0 if dx > 0 else 0.0, 0.5), (x0 if dx > 0 else x1, (y0 + y1) / 2)
        else:
            start, end = (0.5, 0.0 if dy > 0 else 1.0), ((x0 + x1) / 2, y0 if dy > 0 else y1)
        fig.add_artist(ConnectionPatch(xyA=start, coordsA=inset.transAxes, xyB=end, coordsB=ax.transData,
                                       arrowstyle="-|>", mutation_scale=18, color=ACCENT, linewidth=2.0,
                                       path_effects=HALO, zorder=5))
    fig.text(0.01, 1.0 - 0.25 / fig_h, title, fontsize=15, color=TEXT_PRIMARY, ha="left", va="top")
    fig.text(0.01, 1.0 - 0.62 / fig_h, subtitle, fontsize=10, color=TEXT_SECONDARY, ha="left", va="top", linespacing=1.5)
    fig.savefig(path, dpi=min(max(width / panel_w, 150), 400), facecolor=SURFACE)
    plt.close(fig)


def main(args):
    entries, texture_name, image_name, note, (texture_w, texture_h) = select(args)
    full = args.resolution == "full"
    name = args.name or texture_name.replace(" / ", "_") + (f"_{texture_w // 1024:d}k" if full else "")
    at = f", every texture at {texture_w}×{texture_h}" if full else ""
    work_dir = os.path.join(args.out_dir, name)
    os.makedirs(work_dir, exist_ok=True)
    for entry in entries:
        print(f"{entry['title']:40s} {entry['details']:42s} {os.path.relpath(entry['texture'], ROOT)}")
    full = [{"texture": entry["texture"], "output": os.path.join(work_dir, f"{entry['key']}.png"), "percent": 100}
            for entry in entries]
    print(f"Rendering {len(full)} textures with Blender")
    run_blender(args, image_name, full, work_dir, "full")
    renders = [load_render(job["output"]) for job in full]
    height, width = renders[0][0].shape[:2]
    size = int(round(args.zoom_size * height))
    if args.zoom is not None:
        zx, zy = (float(value) for value in args.zoom.split(","))
        x0, y0 = int(round(zx * width - size / 2)), int(round(zy * height - size / 2))
    else:
        x0, y0 = pick_zoom(*renders[0], size)
    box = (x0, y0, x0 + size, y0 + size)
    border = [box[0] / width, box[1] / height, box[2] / width, box[3] / height]
    zoomed = [{"texture": entry["texture"], "output": os.path.join(work_dir, f"{entry['key']}_zoom.png"),
               "percent": args.zoom_percent, "border": border} for entry in entries]
    print(f"Rendering the detail at ({(x0 + size/2)/width:.3f}, {(y0 + size/2)/height:.3f}) of the frame, "
          f"{args.zoom_percent}% resolution")
    run_blender(args, image_name, zoomed, work_dir, "zoom")
    crops = [load_render(job["output"])[0] for job in zoomed]
    # Both figures show the same object crop of every render, the zoom figure widened to hold the inset
    obj = object_box(renders)
    margin = int(round(0.04 * (obj[3] - obj[1])))
    render_region = (obj[0] - margin, obj[1] - margin, obj[2] + margin, obj[3] + margin)
    (ix, iy, side), zoom_region = zoom_layout(obj, box, args.inset_size)
    shift = np.array(zoom_region[:2] * 2)
    scene = os.path.basename(args.blend)
    common = (f"Rendered in Blender from {scene}'s camera. {note}\nSizes are the compressed files; scores are those of "
              "the texture, on the used texels, as in rd_curves.png.")
    paths = [os.path.join(args.out_dir, f"{name}_render.png"), os.path.join(args.out_dir, f"{name}_zoom.png")]
    figure(entries, [crop(rgb, render_region) for rgb, _ in renders], f"{texture_name}: rendered with each texture{at}",
           common, paths[0])
    figure(entries, [crop(rgb, zoom_region) for rgb, _ in renders],
           f"{texture_name}: detail, rendered at {args.zoom_percent / 100:g}x resolution{at}", common, paths[1],
           zoom={"box": tuple(np.array(box) - shift), "inset": (ix - zoom_region[0], iy - zoom_region[1], side),
                 "crops": crops})
    print(f"Saved '{paths[0]}' and '{paths[1]}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a scene with the original, BC/ASTC and Image-GS textures")
    parser.add_argument("blend", type=str, help=".blend file with the textured mesh and its camera")
    parser.add_argument("comparisons", nargs="+", help="Comparison folders of the same texture, e.g. Gaussian and Beta")
    parser.add_argument("--count", type=int, default=None, help="Primitive count (default: the largest one in common)")
    parser.add_argument("--by", type=str, default="psnr", choices=list(METRICS), help="Metric that picks the BC/ASTC result")
    parser.add_argument("--resolution", type=str, default="any", choices=["any", "full"],
                        help="BC/ASTC at any resolution, for equal size (default), or only at the texture's own "
                             "resolution, for equal resolution: then the smallest one if none is as small as Image-GS")
    parser.add_argument("--texture", type=str, default=None,
                        help="Original texture (default: found from the comparison's BC/ASTC folder)")
    parser.add_argument("--zoom", type=str, default=None,
                        help="Center of the detail as x,y fractions of the frame (default: the most detailed area)")
    parser.add_argument("--zoom_size", type=float, default=0.12, help="Side of the detail, as a fraction of the frame height")
    parser.add_argument("--zoom_percent", type=int, default=400, help="Resolution of the detail render, in percent")
    parser.add_argument("--inset_size", type=float, default=0.7, help="Side of the inset, as a fraction of the object's height")
    parser.add_argument("--name", type=str, default=None, help="Output name (default: <texture>_<image>)")
    parser.add_argument("--out_dir", type=str, default=os.path.join(ROOT, "comparisons", "figures"))
    parser.add_argument("--blender", type=str, default="blender")
    main(parser.parse_args())

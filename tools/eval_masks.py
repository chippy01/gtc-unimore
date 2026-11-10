"""
Score empty-area detection against the exact UV mask rasterized from the .obj mesh next to the texture.
"Used lost" is real content wrongly ignored, "empty found" is the share of unused texels that detection catches.
With --csv, every result is also written to a table; running again replaces the rows it computes.
"""
import argparse
import csv
import os
import sys

import yaml

IMAGE_GS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "image-gs")
sys.path.insert(0, IMAGE_GS_DIR)
from utils.image_utils import load_images  # noqa: E402
from utils.mask_utils import EMPTY_MODES, compute_valid_masks, get_uv_mask  # noqa: E402

FIELDS = ["texture", "image", "mode", "tol", "dilate", "used_pct", "ignored_pct", "used_lost_pct", "empty_found_pct"]


def write_csv(path, rows):
    """Merge rows into the table at path, replacing earlier rows of the same texture, image, mode, tol and dilate."""
    key = lambda row: tuple(str(row[field]) for field in FIELDS[:5])  # noqa: E731
    merged = {}
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as file:
            merged = {key(row): row for row in csv.DictReader(file)}
    merged.update({key(row): row for row in rows})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(merged.values(), key=lambda row: (row["texture"], row["image"], row["mode"], float(row["tol"] or 0))))


def main(args):
    with open(os.path.join(IMAGE_GS_DIR, "cfgs", "default.yaml"), "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    images, input_channels, image_fnames, _ = load_images(args.input_path)
    texture = os.path.basename(os.path.dirname(os.path.abspath(args.input_path))) if os.path.isfile(args.input_path) \
        else os.path.basename(os.path.abspath(args.input_path))
    rows = []
    curr_channel = 0
    for num_channels, fname in zip(input_channels, image_fnames):
        image = images[curr_channel:curr_channel+num_channels]
        curr_channel += num_channels
        used = get_uv_mask(args.input_path, fname, image.shape[1:], args.uv_file)
        print(f"'{fname}': {100.0*used.mean():.2f}% of texels used by the mesh")
        for mode in args.mode:
            tols = args.tol
            if tols is None:  # Same defaults as training
                tols = [cfg["empty_detail_tol"] if mode == "blurred" else cfg["empty_tol"]]
            if mode == "uv":
                tols = [0.0]
            for tol in tols:
                valid = compute_valid_masks(image, [num_channels], [fname], args.input_path, mode, tol, args.dilate,
                                            args.uv_file)[0]
                lost = (~valid & used).sum() / max(used.sum(), 1)
                found = (~valid & ~used).sum() / max((~used).sum(), 1)
                label = mode if mode == "uv" else f"{mode} tol {tol:.2f}"
                print(f"  {label:18s} ignored {100.0*(1.0-valid.mean()):6.2f}% | used lost {100.0*lost:6.2f}% | empty found {100.0*found:6.2f}%")
                rows.append({"texture": texture, "image": fname, "mode": mode, "tol": "" if mode == "uv" else tol,
                             "dilate": args.dilate, "used_pct": round(100.0 * used.mean(), 3),
                             "ignored_pct": round(100.0 * (1.0 - valid.mean()), 3),
                             "used_lost_pct": round(100.0 * lost, 3), "empty_found_pct": round(100.0 * found, 3)})
    if args.csv:
        write_csv(args.csv, rows)
        print(f"Saved {len(rows)} rows to '{args.csv}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare empty-area detection with the UV charts of the texture's mesh")
    parser.add_argument("input_path", type=str, help="Texture file or folder, as for main.py --input_path, with one .obj next to it")
    parser.add_argument("--mode", type=str, nargs="+", required=True, choices=[mode for mode in EMPTY_MODES if mode != "none"],
                        help="One or more modes to score, e.g. --mode grey black white blurred uv")
    parser.add_argument("--tol", type=float, nargs="+", default=None,
                        help="One or more tolerances to compare (default: empty_tol or empty_detail_tol from cfgs/default.yaml)")
    parser.add_argument("--dilate", type=int, default=2, help="Same as main.py --empty_dilate")
    parser.add_argument("--uv_file", type=str, default="",
                        help="Ground-truth UV source: an .obj mesh or a mask image (default: the one .obj next to the texture)")
    parser.add_argument("--csv", type=str, default=None, help="Also write the results to this table (rows of a re-run are replaced)")
    main(parser.parse_args())

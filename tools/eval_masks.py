"""
Score empty-area detection against the exact UV mask rasterized from the .obj mesh next to the texture.
"Used lost" is real content wrongly ignored, "empty found" is the share of unused texels that detection catches.
"""
import argparse
import os
import sys

import yaml

IMAGE_GS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "image-gs")
sys.path.insert(0, IMAGE_GS_DIR)
from utils.image_utils import load_images  # noqa: E402
from utils.mask_utils import EMPTY_MODES, compute_valid_masks, get_uv_mask  # noqa: E402


def main(args):
    with open(os.path.join(IMAGE_GS_DIR, "cfgs", "default.yaml"), "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    tols = args.tol
    if tols is None:  # Same defaults as training
        tols = [cfg["empty_detail_tol"] if args.mode == "blurred" else cfg["empty_tol"]]
    if args.mode == "uv":
        tols = [0.0]
    images, input_channels, image_fnames, _ = load_images(args.input_path)
    curr_channel = 0
    for num_channels, fname in zip(input_channels, image_fnames):
        image = images[curr_channel:curr_channel+num_channels]
        curr_channel += num_channels
        used = get_uv_mask(args.input_path, fname, image.shape[1:])
        print(f"'{fname}': {100.0*used.mean():.2f}% of texels used by the mesh")
        for tol in tols:
            valid = compute_valid_masks(image, [num_channels], [fname], args.input_path, args.mode, tol, args.dilate)[0]
            lost = (~valid & used).sum() / max(used.sum(), 1)
            found = (~valid & ~used).sum() / max((~used).sum(), 1)
            label = args.mode if args.mode == "uv" else f"{args.mode} tol {tol:.2f}"
            print(f"  {label:18s} ignored {100.0*(1.0-valid.mean()):6.2f}% | used lost {100.0*lost:6.2f}% | empty found {100.0*found:6.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare empty-area detection with the UV charts of the texture's mesh")
    parser.add_argument("input_path", type=str, help="Texture file or folder, as for main.py --input_path, with one .obj next to it")
    parser.add_argument("--mode", type=str, required=True, choices=[mode for mode in EMPTY_MODES if mode != "none"])
    parser.add_argument("--tol", type=float, nargs="+", default=None,
                        help="One or more tolerances to compare (default: empty_tol or empty_detail_tol from cfgs/default.yaml)")
    parser.add_argument("--dilate", type=int, default=2, help="Same as main.py --empty_dilate")
    main(parser.parse_args())

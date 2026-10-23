import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "image-gs"))
from model import get_pixel_scale, render_gaussians  # noqa: E402
from utils.gtc_utils import load_gtc  # noqa: E402
from utils.image_utils import save_image, separate_image_channels  # noqa: E402


def main(args):
    gaussians = load_gtc(args.gtc_file)
    xy, scale, rot, feat = [torch.from_numpy(param).to(args.device) for param in
                            (gaussians.xy, gaussians.scale, gaussians.rot, gaussians.feat)]
    img_h = gaussians.img_h if args.render_height is None else args.render_height
    img_w = round((float(img_h)/gaussians.img_h)*gaussians.img_w)
    block_h, block_w = 16, 16  # Warning: Must match hardcoded value in CUDA kernel, modify with caution
    tile_bounds = ((img_w + block_w - 1) // block_w, (img_h + block_h - 1) // block_h, 1)
    scale = get_pixel_scale(scale, inverse_scale=gaussians.inverse_scale, upsample_ratio=float(img_h)/gaussians.img_h)
    render_args = (xy, scale, rot, feat, img_h, img_w, tile_bounds, block_h, block_w)
    render_kwargs = dict(bits=gaussians.bits, enable_tiles=gaussians.tiles, enable_topk_norm=gaussians.topk_norm)
    with torch.no_grad():
        num_prep_runs = 2
        for _ in range(num_prep_runs):
            render_gaussians(*render_args, **render_kwargs, benchmark=True)
        images, render_time = render_gaussians(*render_args, **render_kwargs)
    out_dir = os.path.dirname(os.path.abspath(args.gtc_file)) if args.out_dir is None else args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    # Images keep their original names, so they can replace the original textures of a mesh
    for image, fname in zip(separate_image_channels(images=images, input_channels=gaussians.input_channels), gaussians.image_fnames):
        path = os.path.join(out_dir, f"{fname}.{args.format}")
        if os.path.exists(path) and not args.overwrite:
            raise FileExistsError(f"'{path}' already exists, pass --overwrite to replace it")
        save_image(image, path, gamma=gaussians.gamma)
        print(f"Saved '{path}'")
    print(f"Decoded {len(gaussians.xy):d} Gaussians ({os.path.getsize(args.gtc_file)/1e3:.2f} KB) "
          f"at resolution ({img_h:d}, {img_w:d}) in {render_time:.6f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a .gtc file written by main.py into image files")
    parser.add_argument("gtc_file", type=str, help="Path to a .gtc file")
    parser.add_argument("--render_height", type=int, default=None, help="Image height (default: optimization resolution)")
    parser.add_argument("--format", type=str, default="png", choices=["png", "jpg"], help="png saves 16-bit lossless files")
    parser.add_argument("--out_dir", type=str, default=None, help="Output folder (default: folder of the .gtc file)")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing files with the same name")
    parser.add_argument("--device", type=str, default="cuda:0")
    main(parser.parse_args())

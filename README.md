# gtc-unimore
Gaussian Texture Compression experiment

Textures are compressed into a set of colored 2D Gaussians with [Image-GS](https://github.com/NYU-ICL/image-gs) (Zhang et al., SIGGRAPH 2025). The engine is copied into `image-gs/` from upstream commit `0308836` and extended so that it can **ignore the empty areas of texture atlases**: the texels that no face of the mesh uses, so no Gaussians are spent on them.

## Repository layout
| Path | Content |
|---|---|
| `image-gs/` | Image-GS engine with the changes described below |
| `image-gs/utils/mask_utils.py` | Empty-area detection (new) |
| `image-gs/utils/gtc_utils.py` | Compressed `.gtc` file format (new) |
| `tools/` | Standalone scripts: comparison against BC1/BC7/ASTC, `.gtc` decoder, mask accuracy check (new), see [Tools](#tools) |
| `non_gaussian_compressions/` | BC1, BC7 and ASTC files written by the comparison. Not tracked by git |
| `comparisons/` | Metrics tables and plots written by the comparison |
| `image-gs/cfgs/default.yaml` | All settings and their defaults, every key is also a command-line flag |
| `RWT78/`, `RWT79/`, `RWT81/`, `RWT85/` | Test textures with their meshes, from the *Real-World Textured Things* (RWTT) dataset. Not tracked by git, copy them in by hand |

## Setup
Tested on Fedora 43 with an RTX 3070 Ti. The CUDA toolkit and a compatible gcc are installed inside the conda environment, so the system only needs the NVIDIA driver.

```bash
conda create -n image-gs -c conda-forge --override-channels python=3.11 -y
conda activate image-gs
conda install -c conda-forge --override-channels gxx_linux-64=13 -y
conda install -c nvidia/label/cuda-12.4.1 -c conda-forge --override-channels cuda-toolkit=12.4.1 -y
conda env config vars set CUDA_HOME=$CONDA_PREFIX CPATH=$CONDA_PREFIX/targets/x86_64-linux/include TORCH_CUDA_ARCH_LIST=8.6
conda deactivate && conda activate image-gs

pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
pip install flip-evaluator lpips==0.1.4 matplotlib==3.9.2 numpy==2.0.2 opencv-python==4.12.0.88 \
    pytorch-msssim==1.0.0 scikit-image==0.24.0 scipy==1.13.1 torchmetrics==1.5.2
pip install git+https://github.com/rahul-goel/fused-ssim/ --no-build-isolation
cd image-gs/gsplat
pip install -e ".[dev]" --no-build-isolation --config-settings editable_mode=compat
cd ..
python -c "import torch, gsplat, fused_ssim; print(torch.cuda.is_available())"   # should print True
# Only for the comparison against BC1, BC7 and ASTC
pip install --only-binary=:all: etcpak==0.9.15 texture2ddecoder==1.0.6 astc-encoder-py==0.1.12
```

- `TORCH_CUDA_ARCH_LIST=8.6` matches RTX 30xx GPUs. Use e.g. `8.9` for RTX 40xx.
- `editable_mode=compat` is required. Without it, running from `image-gs/` picks up the outer `gsplat/` folder and fails with `cannot import name 'rasterize_gaussians_no_tiles'`.
- The first run downloads a 233 MB AlexNet model into `image-gs/models/` for the LPIPS metric.

## Running
Run `main.py` from `image-gs/`, because it loads its config and writes its results relative to that folder. `--data_root=..` makes `--input_path` relative to the repository root. The scripts in `tools/` work from any folder.

```bash
cd image-gs
# Optimize 10000 Gaussians with 16-bit parameters
python main.py --data_root=.. --input_path="RWT79" --exp_name="rwt79" --num_gaussians=10000 --quantize
# Render the result at 4096 px height (repeat the training flags so it finds the same run folder)
python main.py --data_root=.. --input_path="RWT79" --exp_name="rwt79" --num_gaussians=10000 --quantize --eval --render_height=4096
```

- `--input_path` is an image file or a folder. A folder loads **all** its images as one multi-channel stack, which is meant for material maps of the same surface (e.g. diffuse + normal).
- `--num_gaussians` sets the compressed size. With `--quantize`, an RGB Gaussian takes 16 bytes, so 10000 Gaussians take 160 KB.
- `--max_steps` below 5000 is raised to 5000, because progressive optimization needs at least that many. Training also stops early once quality stops improving.

## Ignoring empty texture areas
`--empty_mode` tells the software how the empty areas look in the texture:

| `--empty_mode` | Empty areas look like | How they are detected |
|---|---|---|
| `none` (default) | — | Nothing is ignored, as in upstream Image-GS |
| `grey` | Flat grey fill (RWT78) | Within `--empty_tol` of the most common grey level |
| `black` | Flat black fill (RWT85) | Within `--empty_tol` of black |
| `white` | Flat white fill | Within `--empty_tol` of white |
| `blurred` | Smeared, blurred fill (RWT79, RWT81) | Local fine detail below `--empty_detail_tol` |
| `uv` | Anything | Rasterizes the UV charts of the `.obj` in the texture's folder (exact) |

| Option | Default | Meaning |
|---|---|---|
| `--empty_tol` | `2.0` | Flat fills: maximum per-channel distance from the fill colour, in 8-bit levels |
| `--empty_detail_tol` | `0.25` | Blurred fill: maximum local detail, in 8-bit levels. Depends strongly on the texture |
| `--empty_dilate` | `2` | Pixels of empty texels kept around used areas, so texture filtering at chart borders stays correct |
| `--empty_eval_only` | off | Detect empty areas for the metrics only, but still optimize every pixel (baseline) |

Empty texels are ignored by the loss (L1, L2, SSIM), by the initial placement of Gaussians, by error-guided Gaussian addition, and by the reported metrics.

```bash
# Exact mask from the mesh (recommended for the RWT textures)
python main.py --data_root=.. --input_path="RWT79" --exp_name="rwt79" --num_gaussians=10000 --quantize --empty_mode=uv
# Baseline for comparison: same masked metrics, but every pixel is optimized
python main.py --data_root=.. --input_path="RWT79" --exp_name="rwt79" --num_gaussians=10000 --quantize --empty_mode=uv --empty_eval_only
# Detection from the texture alone
python main.py --data_root=.. --input_path="RWT78/tex_0.png" --exp_name="rwt78_0" --num_gaussians=10000 --quantize --empty_mode=grey
python main.py --data_root=.. --input_path="RWT85" --exp_name="rwt85" --num_gaussians=10000 --quantize --empty_mode=black
python main.py --data_root=.. --input_path="RWT79" --exp_name="rwt79" --num_gaussians=10000 --quantize --empty_mode=blurred --empty_detail_tol=0.5
```

Each run saves `mask_res-<H>x<W>.jpg` (white = used, black = ignored) and logs the share of ignored texels, so check both before trusting a result.

### Which mode for which texture
Measured against masks rasterized from the meshes. "Used lost" is real content wrongly ignored, which then renders as garbage.

| Texture | Mode | Used lost | Empty found | Note |
|---|---|---|---|---|
| RWT78 `tex_0` | `grey` | 0% | 47% | The paper patches hold image content but no face uses them, so only `uv` finds them |
| RWT79 | `blurred` 0.5 | 0% | 54% | 0.25 finds almost nothing, higher values start losing content |
| RWT81 | `blurred` 0.25 | 1.7% | 80% | Real content is nearly as smooth as the fill, 0.5 loses about a third of it |
| RWT85 | `black` | 44% | 57% | Unphotographed faces are pure black, so they are ignored too and render as smeared colour |
| All | `uv` | 0% | 96–99.6% | Exact, the rest is the `--empty_dilate` margin |

To reproduce these numbers or test other tolerances, use `tools/eval_masks.py` (see [Tools](#tools)).

### Notes
- **RWT78** holds four separate atlases (`tex_0.png` … `tex_3.png`), not material maps of one surface. Run each file on its own instead of the folder.
- **Metrics:** with a mask, PSNR, SSIM, LPIPS and FLIP count used texels only. MS-SSIM cannot be masked, so it is computed with empty texels set to the target, which flatters it slightly. Compare masked runs with `--empty_eval_only` runs, not with runs that use `--empty_mode=none`.
- **Gaussians in empty areas:** a few percent of Gaussians can drift fully into empty space during training. Nothing pulls them back there, so they are wasted.

## The compressed file
Every training run ends by writing `gaussians.gtc` into its run folder. It holds the Gaussians, with each parameter bit-packed at its `--*_bits` precision, plus a header of about 70 bytes (resolution, image names and channels, settings). Its size is the real compressed size: 160.07 KB for 10000 Gaussians at 16 bits. For runs trained before this feature, repeat the training command with `--eval` to write it.

`tools/decode.py` renders a `.gtc` file back into images. It needs only that file, no checkpoint and no original texture:

```bash
cd image-gs
# PNG at the optimization resolution, next to the .gtc file
python ../tools/decode.py results/rwt79/<run-folder>/gaussians.gtc
# Any resolution, into another folder
python ../tools/decode.py results/rwt79/<run-folder>/gaussians.gtc --render_height=8192 --out_dir=../view/rwt79
```

- Images are named after the original textures (e.g. `ufo_crashste_brno_DiffuseMap.png`), so they can be placed next to a copy of the mesh's `.obj` and `.mtl` to view the compressed texture in Blender.
- Existing files are only replaced with `--overwrite`. Never combine `--overwrite` with an `--out_dir` that holds the original textures.
- `--format=png` (default) writes lossless 16-bit PNGs, `--format=jpg` writes 8-bit JPEGs.
- The decoded images match the `--eval` renders. `render_res-*` is saved just before training quantizes the final checkpoint, so it can differ slightly (well under one 8-bit level on average). The final metrics in the log are measured after that step, so they describe the `.gtc` file.

## Comparing against BC1, BC7 and ASTC
`tools/compare_compression.sh` compresses a texture with Image-GS and with the standard GPU texture formats, scores every result on the same mask, and plots quality against size. Run it from any folder (it activates the conda environment itself):

```bash
tools/compare_compression.sh RWT79                         # defaults: 2500 to 40000 Gaussians, 'uv' mask
tools/compare_compression.sh -g "5000 20000" -m black RWT85
tools/compare_compression.sh -m blurred -t 0.5 RWT79/ufo_crashste_brno_DiffuseMap.png
```

| Option | Default | Meaning |
|---|---|---|
| `-g` | `"2500 5000 10000 20000 40000"` | Gaussian counts, one Image-GS training run each |
| `-s` | `"1 2 4 8 16"` | Downscaling factors for BC/ASTC, `4` compresses the texture at 1/4 of its width and height |
| `-m` | `uv` | Empty-area mode, used for training and for scoring every method |
| `-t` | `main.py` default | Tolerance for `-m`: `--empty_detail_tol` for `blurred`, `--empty_tol` otherwise |

Given a folder, every image in it is compared on its own (RWT78 gives four comparisons). An Image-GS run at 4096² takes about 7 minutes or more, and more Gaussians take longer. The BC/ASTC encodes take under a minute, and scoring about one more. Runs that already have a `gaussians.gtc` are reused, so after an interruption only the missing counts are trained. Delete a run folder to retrain it.

| Folder | Content |
|---|---|
| `image-gs/results/compare/<texture>/<image>_empty-<mode>/` | One Image-GS run per Gaussian count |
| `non_gaussian_compressions/<texture>/<image>/` | `<codec>_<W>x<H>.dds` / `.astc` files, their decoded `.png`, and `baselines.csv` |
| `comparisons/<texture>/<image>/` | `metrics.csv` (every result), `rd_curves.png` (PSNR, SSIM, LPIPS, FLIP against bits per texel) and `sizes.png` (file size of every full-resolution result, next to the uncompressed size and the original file) |

How results are measured:
- **Size** is the compressed file on disk (`.gtc`, `.dds`, `.astc`, headers included), as bits per texel of the original resolution. No method stores mipmaps.
- **Quality:** every method is decoded at the original resolution and scored against the original on the mask from `-m`, with the same metrics as training.
- **Lower resolutions:** at full resolution, BC1 is 4 bpp, BC7 8 bpp and ASTC 8 to 0.89 bpp, far above Image-GS (about 0.02 to 0.3 bpp). A smaller texture is what an engine would ship at those sizes. Downscaling uses area filtering, like mipmaps, and decoded results are upscaled bilinearly, like GPU magnification.
- **Encoders:** BC1 and BC7 by `etcpak` (BC7 through its bc7enc encoder at default settings), ASTC by ARM's `astc-encoder` at its *thorough* preset. Everything is decoded by the independent `texture2ddecoder`, and the `.dds` files also open correctly in Pillow. `etcpak`'s BC1 encoder is tuned for speed, so it scores about 1 dB below the best BC1 encoders.
- **Very large textures** such as RWT81 (8196²) may not fit LPIPS and FLIP into 8 GB of GPU memory. Those values are then left empty, with a warning.

## Tools
Standalone scripts in `tools/`. They need the `image-gs` conda environment and work from any folder. The examples below run from the repository root.

| Script | Purpose |
|---|---|
| `compare_compression.sh` | Full comparison against BC1, BC7 and ASTC, see [above](#comparing-against-bc1-bc7-and-astc) |
| `baseline_codecs.py` | BC1, BC7 and ASTC encoding at several resolutions (called by `compare_compression.sh`) |
| `plot_compression.py` | Scoring and plotting of all methods (called by `compare_compression.sh`) |
| `decode.py` | Render a `.gtc` file into images, see [The compressed file](#the-compressed-file) |
| `eval_masks.py` | Score empty-area detection against the exact UV mask rasterized from the texture's mesh |

```bash
# One mode, with its default tolerance from image-gs/cfgs/default.yaml
python tools/eval_masks.py RWT85 --mode black
# Several tolerances in one go
python tools/eval_masks.py RWT79 --mode blurred --tol 0.25 0.5 0.75
```

For each image, `eval_masks.py` prints the share of ignored texels, of used texels wrongly ignored ("used lost") and of unused texels caught ("empty found"). `--dilate` matches `main.py --empty_dilate`. The texture's folder must contain exactly one `.obj` file.

## Outputs
Each run writes to `image-gs/results/<exp_name>/<settings>/`. The settings part of the name includes the empty-area options (e.g. `..._empty-uv-dil-2_...`), so runs with different settings never overwrite each other. **Rerunning with identical settings deletes the previous run folder.**

| File | Content |
|---|---|
| `gaussians.gtc` | Compressed texture, input for `tools/decode.py` |
| `render_res-<H>x<W>.jpg` | Final reconstruction (`.png` with `--save_image_format=png`) |
| `gt_res-<H>x<W>.jpg`, `mask_res-<H>x<W>.jpg` | Target texture and empty-area mask |
| `log_train.txt` | Size, compression rate, metrics during training |
| `checkpoints/ckpt_step-<N>.pt` | Optimized Gaussians and final metrics |
| `eval/` | Renders made with `--eval` (which also rewrites `gaussians.gtc`) |

## Credits
- Image-GS: Zhang et al., *Image-GS: Content-Adaptive Image Representation via 2D Gaussians*, SIGGRAPH 2025, MIT license (see `image-gs/README.md` and `image-gs/LICENSE`).
- Textures: RWTT dataset. Each folder's `metadata.json` names the original author and license. RWT81 is non-commercial.

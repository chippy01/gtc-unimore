#!/usr/bin/env bash
# Compress a texture with Image-GS, BC1, BC7 and ASTC, then score every result on the same mask and plot quality
# against size.
#   Image-GS runs  -> image-gs/results/compare/<texture>/<image>_empty-<mode>[_<label>]/
#   BC1/BC7/ASTC   -> non_gaussian_compressions/<texture>/<image>/
#   metrics + plot -> comparisons/<texture>/<image>[_<label>]/
#   with -r, textures larger than the cap are scaled down once into capped_textures/<max>/<texture>/, and <image>
#   becomes <image>_max<max>
# Image-GS runs that already have a gaussians.gtc are reused, so an interrupted comparison can be restarted.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GAUSSIANS="2500 5000 10000 20000 40000"
SCALES="1 2 4 8 16"
EMPTY_MODE="uv"
TOL=""
UV_FILE=""
EXTRA=""
LABEL=""
MAX_RES=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [options] <texture folder or image file>

Every image in the folder (e.g. RWT78/tex_0.png ... tex_3.png) is compared on its own.

Options:
  -g "N1 N2 ..."  Gaussian counts for Image-GS (default: "$GAUSSIANS")
  -s "S1 S2 ..."  Downscaling factors for BC/ASTC, 4 = 1/4 width and height (default: "$SCALES")
  -m MODE         Empty-area mode for training and scoring: none, grey, black, white, blurred, uv (default: $EMPTY_MODE)
  -t TOL          Tolerance for MODE: --empty_detail_tol for blurred, --empty_tol otherwise (default: main.py's)
  -u FILE         UV source for MODE uv: an .obj mesh or a mask image, white = used (default: the .obj next to the texture)
  -x "ARGS"       Extra main.py options for the Image-GS runs, space-separated (e.g. "--pos_bits=12 --feat_bits=10")
  -n LABEL        Name that keeps these runs apart from others on the same texture (default: derived from -x)
  -r MAX          Scale textures whose longest side exceeds MAX pixels down to MAX before anything else, to fit GPU
                  memory (e.g. 4096; training an 8196x8196 texture needs more than 8 GB). Default: no limit
  -h              Show this help

Example: $(basename "$0") -g "5000 20000" RWT79
EOF
}

# Prints the image to work on: the original, or a copy whose longest side is MAX_RES. All steps read the same 8-bit
# copy, so training, BC/ASTC and scoring see exactly the same texture
capped_image() {
    python - "$1" "$MAX_RES" "$ROOT/capped_textures/$MAX_RES/$TEXTURE" <<'EOF'
import os
import sys

import cv2

path, max_res, out_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
height, width = image.shape[:2]
if max(height, width) <= max_res:
    print(path)
    sys.exit()
scale = max_res / max(height, width)
size = (max(1, round(width * scale)), max(1, round(height * scale)))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, os.path.basename(path))  # Same name, so the .mtl still matches the texture
if not os.path.exists(out_path) or os.path.getmtime(out_path) < os.path.getmtime(path):
    cv2.imwrite(out_path, cv2.resize(image, size, interpolation=cv2.INTER_AREA))  # Area filtering, like mipmaps
# Link the mesh files next to the copy, so that empty_mode "uv" still finds the .obj next to the texture
source_dir = os.path.dirname(path)
for name in os.listdir(source_dir):
    if name.lower().endswith((".obj", ".mtl")) and not os.path.lexists(os.path.join(out_dir, name)):
        os.symlink(os.path.join(source_dir, name), os.path.join(out_dir, name))
print(f"--- {os.path.basename(path)} is {width}x{height}, using a {size[0]}x{size[1]} copy: {out_path}", file=sys.stderr)
print(out_path)
EOF
}

while getopts "g:s:m:t:u:x:n:r:h" opt; do
    case "$opt" in
        g) GAUSSIANS="$OPTARG" ;;
        s) SCALES="$OPTARG" ;;
        m) EMPTY_MODE="$OPTARG" ;;
        t) TOL="$OPTARG" ;;
        u) UV_FILE="$(realpath "$OPTARG")" ;;
        x) EXTRA="$OPTARG" ;;
        n) LABEL="$OPTARG" ;;
        r) MAX_RES="$OPTARG" ;;
        h) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
done
read -r -a EXTRA_ARGS <<< "$EXTRA"
# Runs are reused by Gaussian count, so other main.py options need their own folder: "--pos_bits=12" -> "pos_bits-12"
if [[ -z "$LABEL" && -n "$EXTRA" ]]; then
    LABEL="$(echo "$EXTRA" | sed -e 's/--//g' -e 's/=/-/g' -e 's/[^A-Za-z0-9_.-]\+/_/g' -e 's/^_//' -e 's/_$//')"
fi
SUFFIX="${LABEL:+_$LABEL}"
if [[ -n "$MAX_RES" && ! "$MAX_RES" =~ ^[1-9][0-9]*$ ]]; then
    echo "-r needs a positive whole number of pixels, got '$MAX_RES'" >&2
    exit 1
fi
shift $((OPTIND - 1))
if [[ $# -ne 1 ]]; then
    usage
    exit 1
fi
TARGET="$(realpath "$1")"

if [[ "${CONDA_DEFAULT_ENV:-}" != "image-gs" ]]; then
    set +u  # conda's activation scripts read unset variables
    eval "$(conda shell.bash hook)"
    conda activate image-gs
    set -u
fi
if ! python -c "import etcpak, texture2ddecoder, astc_encoder" 2>/dev/null; then
    echo "Missing BC/ASTC encoders, install them with:" >&2
    echo "  pip install --only-binary=:all: etcpak==0.9.15 texture2ddecoder==1.0.6 astc-encoder-py==0.1.12" >&2
    exit 1
fi

if [[ -d "$TARGET" ]]; then
    TEXTURE="$(basename "$TARGET")"
    mapfile -t IMAGES < <(find "$TARGET" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \
        -o -iname '*.tif' -o -iname '*.tiff' -o -iname '*.exr' \) | sort)
else
    TEXTURE="$(basename "$(dirname "$TARGET")")"
    IMAGES=("$TARGET")
fi
if [[ ${#IMAGES[@]} -eq 0 ]]; then
    echo "No image found in '$TARGET'" >&2
    exit 1
fi

MASK_ARGS=(--empty_mode="$EMPTY_MODE")
if [[ -n "$TOL" ]]; then
    if [[ "$EMPTY_MODE" == "blurred" ]]; then
        MASK_ARGS+=(--empty_detail_tol="$TOL")
    else
        MASK_ARGS+=(--empty_tol="$TOL")
    fi
fi
if [[ -n "$UV_FILE" ]]; then
    MASK_ARGS+=(--empty_uv_file="$UV_FILE")
fi

for SOURCE in "${IMAGES[@]}"; do
    IMAGE="$SOURCE"
    NAME="$(basename "${SOURCE%.*}")"
    if [[ -n "$MAX_RES" ]]; then
        IMAGE="$(capped_image "$SOURCE")"
        if [[ "$IMAGE" != "$SOURCE" ]]; then
            NAME="${NAME}_max$MAX_RES"  # Keeps capped results apart from full-resolution ones
        fi
    fi
    EXP_NAME="compare/$TEXTURE/${NAME}_empty-$EMPTY_MODE$SUFFIX"
    BASELINE_DIR="$ROOT/non_gaussian_compressions/$TEXTURE/$NAME"
    echo "=== $TEXTURE / $NAME$SUFFIX"

    for N in $GAUSSIANS; do
        # Beta Splatting runs ("_beta_" in the folder name, from the feat_beta_splatting branch) are not reused
        if compgen -G "$ROOT/image-gs/results/$EXP_NAME/num-${N}_*/gaussians.gtc" | grep -v "_beta_" > /dev/null; then
            echo "--- Image-GS, $N Gaussians: reusing the existing run"
            continue
        fi
        echo "--- Image-GS, $N Gaussians"
        # main.py must run from image-gs/; an absolute --input_path ignores --data_root
        (cd "$ROOT/image-gs" && python main.py --input_path="$IMAGE" --exp_name="$EXP_NAME" --num_gaussians="$N" \
            --quantize "${MASK_ARGS[@]}" "${EXTRA_ARGS[@]}")
    done

    echo "--- BC1, BC7, ASTC"
    # shellcheck disable=SC2086
    python "$ROOT/tools/baseline_codecs.py" "$IMAGE" --out_dir "$BASELINE_DIR" --scales $SCALES

    echo "--- Scoring and plotting"
    python "$ROOT/tools/plot_compression.py" "$IMAGE" --baselines "$BASELINE_DIR" \
        --gaussians "$ROOT/image-gs/results/$EXP_NAME" --out_dir "$ROOT/comparisons/$TEXTURE/$NAME$SUFFIX" "${MASK_ARGS[@]}"
done

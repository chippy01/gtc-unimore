#!/usr/bin/env bash
# Compress a texture with Image-GS, BC1, BC7 and ASTC, then score every result on the same mask and plot quality
# against size.
#   Image-GS runs  -> image-gs/results/compare/<texture>/<image>_empty-<mode>/
#   BC1/BC7/ASTC   -> non_gaussian_compressions/<texture>/<image>/
#   metrics + plot -> comparisons/<texture>/<image>/
# Image-GS runs that already have a gaussians.gtc are reused, so an interrupted comparison can be restarted.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GAUSSIANS="2500 5000 10000 20000 40000"
SCALES="1 2 4 8 16"
EMPTY_MODE="uv"
TOL=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [options] <texture folder or image file>

Every image in the folder (e.g. RWT78/tex_0.png ... tex_3.png) is compared on its own.

Options:
  -g "N1 N2 ..."  Gaussian counts for Image-GS (default: "$GAUSSIANS")
  -s "S1 S2 ..."  Downscaling factors for BC/ASTC, 4 = 1/4 width and height (default: "$SCALES")
  -m MODE         Empty-area mode for training and scoring: none, grey, black, white, blurred, uv (default: $EMPTY_MODE)
  -t TOL          Tolerance for MODE: --empty_detail_tol for blurred, --empty_tol otherwise (default: main.py's)
  -h              Show this help

Example: $(basename "$0") -g "5000 20000" RWT79
EOF
}

while getopts "g:s:m:t:h" opt; do
    case "$opt" in
        g) GAUSSIANS="$OPTARG" ;;
        s) SCALES="$OPTARG" ;;
        m) EMPTY_MODE="$OPTARG" ;;
        t) TOL="$OPTARG" ;;
        h) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
done
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

for IMAGE in "${IMAGES[@]}"; do
    NAME="$(basename "${IMAGE%.*}")"
    EXP_NAME="compare/$TEXTURE/${NAME}_empty-$EMPTY_MODE"
    BASELINE_DIR="$ROOT/non_gaussian_compressions/$TEXTURE/$NAME"
    echo "=== $TEXTURE / $NAME"

    for N in $GAUSSIANS; do
        if compgen -G "$ROOT/image-gs/results/$EXP_NAME/num-${N}_*/gaussians.gtc" > /dev/null; then
            echo "--- Image-GS, $N Gaussians: reusing the existing run"
            continue
        fi
        echo "--- Image-GS, $N Gaussians"
        # main.py must run from image-gs/; an absolute --input_path ignores --data_root
        (cd "$ROOT/image-gs" && python main.py --input_path="$IMAGE" --exp_name="$EXP_NAME" --num_gaussians="$N" \
            --quantize "${MASK_ARGS[@]}")
    done

    echo "--- BC1, BC7, ASTC"
    # shellcheck disable=SC2086
    python "$ROOT/tools/baseline_codecs.py" "$IMAGE" --out_dir "$BASELINE_DIR" --scales $SCALES

    echo "--- Scoring and plotting"
    python "$ROOT/tools/plot_compression.py" "$IMAGE" --baselines "$BASELINE_DIR" \
        --gaussians "$ROOT/image-gs/results/$EXP_NAME" --out_dir "$ROOT/comparisons/$TEXTURE/$NAME" "${MASK_ARGS[@]}"
done

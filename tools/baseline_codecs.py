"""
Compress a texture with GPU block formats (BC1, BC7, ASTC) at several resolutions, as baselines for Image-GS.
For each codec and scale, writes the compressed file (.dds or .astc), its decoded image (.png, at the compressed
resolution) and a manifest (baselines.csv) that tools/plot_compression.py reads.
"""
import argparse
import csv
import os
import struct
from time import perf_counter

import cv2
import etcpak
import numpy as np
import texture2ddecoder
from astc_encoder import ASTCConfig, ASTCContext, ASTCImage, ASTCProfile, ASTCQualityPreset, ASTCSwizzle, ASTCType

# Codec name: (block width, block height)
CODECS = {"bc1": (4, 4), "bc7": (4, 4), "astc4x4": (4, 4), "astc6x6": (6, 6), "astc8x8": (8, 8), "astc12x12": (12, 12)}
DXGI_FORMAT_BC7_UNORM = 98


def load_rgb8(path):
    """Load an image as [H, W, 3] uint8 RGB, since BC and ASTC (LDR) store 8 bits per channel."""
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Cannot read image '{path}'")
    if image.dtype == np.uint16:
        image = np.round(image / 257.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    return np.ascontiguousarray(image[..., 2::-1])  # BGR(A) -> RGB, alpha dropped


def to_rgba_bytes(image):
    return np.dstack([image, np.full(image.shape[:2], 255, dtype=np.uint8)]).tobytes()


def from_bgra_bytes(data, width, height):
    """texture2ddecoder returns BGRA."""
    return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)[..., 2::-1]


def pad_to_blocks(image, block_w, block_h):
    pad_h, pad_w = (-image.shape[0]) % block_h, (-image.shape[1]) % block_w
    return np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def encode_bc(image, codec):
    """Returns (compressed blocks, decoded [H, W, 3] image). BC needs whole 4x4 blocks, so edges are replicated."""
    height, width = image.shape[:2]
    padded = pad_to_blocks(image, 4, 4)
    pad_h, pad_w = padded.shape[:2]
    if codec == "bc1":
        data = etcpak.compress_bc1(to_rgba_bytes(padded), pad_w, pad_h)
        decoded = texture2ddecoder.decode_bc1(data, pad_w, pad_h)
    else:
        data = etcpak.compress_bc7(to_rgba_bytes(padded), pad_w, pad_h, etcpak.BC7CompressBlockParams())
        decoded = texture2ddecoder.decode_bc7(data, pad_w, pad_h)
    return data, from_bgra_bytes(decoded, pad_w, pad_h)[:height, :width]


def encode_astc(image, block_w, block_h, threads):
    height, width = image.shape[:2]
    config = ASTCConfig(ASTCProfile.LDR, block_w, block_h, 1, ASTCQualityPreset.THOROUGH)
    context = ASTCContext(config, threads=threads)
    data = context.compress(ASTCImage(ASTCType.U8, width, height, 1, to_rgba_bytes(image)), ASTCSwizzle.from_str("RGBA"))
    decoded = texture2ddecoder.decode_astc(data, width, height, block_w, block_h)
    return data, from_bgra_bytes(decoded, width, height)


def write_dds(path, data, width, height, codec):
    """Minimal DDS file: BC1 as DXT1, BC7 through the DX10 extension header."""
    flags = 0x1 | 0x2 | 0x4 | 0x1000 | 0x80000  # CAPS | HEIGHT | WIDTH | PIXELFORMAT | LINEARSIZE
    four_cc = b"DXT1" if codec == "bc1" else b"DX10"
    header = struct.pack("<4s7I11I2I4s5I5I", b"DDS ", 124, flags, height, width, len(data), 0, 1, *([0] * 11),
                         32, 0x4, four_cc, 0, 0, 0, 0, 0, 0x1000, 0, 0, 0, 0)
    if codec == "bc7":
        header += struct.pack("<5I", DXGI_FORMAT_BC7_UNORM, 3, 0, 1, 0)  # 2D texture, array size 1
    with open(path, "wb") as file:
        file.write(header + data)


def write_astc(path, data, width, height, block_w, block_h):
    """ARM .astc file: 16-byte header (magic, block size, 24-bit image size) followed by the blocks."""
    header = bytes([0x13, 0xAB, 0xA1, 0x5C, block_w, block_h, 1])
    header += width.to_bytes(3, "little") + height.to_bytes(3, "little") + (1).to_bytes(3, "little")
    with open(path, "wb") as file:
        file.write(header + data)


def main(args):
    original = load_rgb8(args.image)
    height, width = original.shape[:2]
    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for scale in args.scales:
        scaled_w, scaled_h = max(1, round(width / scale)), max(1, round(height / scale))
        # Area filtering matches how mipmaps are made
        image = original if scale == 1 else cv2.resize(original, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        for codec in args.codecs:
            block_w, block_h = CODECS[codec]
            begin = perf_counter()
            if codec.startswith("bc"):
                data, decoded = encode_bc(image, codec)
            else:
                data, decoded = encode_astc(image, block_w, block_h, args.threads)
            encode_time = perf_counter() - begin
            name = f"{codec}_{scaled_w:d}x{scaled_h:d}"
            compressed_path = os.path.join(args.out_dir, name + (".dds" if codec.startswith("bc") else ".astc"))
            if codec.startswith("bc"):
                write_dds(compressed_path, data, scaled_w, scaled_h, codec)
            else:
                write_astc(compressed_path, data, scaled_w, scaled_h, block_w, block_h)
            decoded_path = os.path.join(args.out_dir, name + ".png")
            cv2.imwrite(decoded_path, decoded[..., ::-1])
            rows.append({"codec": codec, "scale": scale, "width": scaled_w, "height": scaled_h,
                         "compressed_file": os.path.basename(compressed_path), "decoded_file": os.path.basename(decoded_path),
                         "bytes": os.path.getsize(compressed_path), "encode_seconds": round(encode_time, 3)})
            print(f"{name:22s} {rows[-1]['bytes']/1e3:10.2f} KB  {encode_time:6.2f} s")
    with open(os.path.join(args.out_dir, "baselines.csv"), "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress a texture with BC1, BC7 and ASTC at several resolutions")
    parser.add_argument("image", type=str, help="Texture image file")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--codecs", type=str, nargs="+", default=list(CODECS), choices=list(CODECS))
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 2, 4, 8, 16],
                        help="Downscaling factors, e.g. 4 compresses the texture at 1/4 of its width and height")
    parser.add_argument("--threads", type=int, default=os.cpu_count(), help="ASTC encoder threads")
    main(parser.parse_args())

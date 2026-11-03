import struct
from dataclasses import dataclass

import numpy as np

# File layout (little-endian):
#   header       magic, image height, image width, number of Gaussians, flags, 4 bit precisions, gamma
#   images       count, then per image: number of channels, name length, UTF-8 name
#   parameters   xy [N, 2], scale [N, 2], rot [N, 1], feat [N, C], each as min/max (float32) followed by
#                its quantization codes bit-packed at its precision, or as raw float32 values when not quantized
MAGIC = b"GTC1"
HEADER = struct.Struct("<4sIIIB4Bf")
RANGE = struct.Struct("<2f")
FLAG_QUANTIZED, FLAG_INVERSE_SCALE, FLAG_TOPK_NORM, FLAG_TILES = 1, 2, 4, 8


@dataclass
class GaussianImage:
    xy: np.ndarray  # [N, 2] positions, image spans [0, 1]
    scale: np.ndarray  # [N, 2] scale parameters as optimized (inverse scales unless inverse_scale is False)
    rot: np.ndarray  # [N, 1] rotation angles
    feat: np.ndarray  # [N, C] features of all stacked images
    bits: tuple  # Bit precisions of (position, scale, rotation, feature), None for float32 parameters
    img_h: int  # Optimization resolution, Gaussian scales are measured in its pixels
    img_w: int
    input_channels: list  # Number of channels of each stacked image
    image_fnames: list
    gamma: float
    inverse_scale: bool
    topk_norm: bool
    tiles: bool


def save_gtc(path, gaussians):
    """
    Write Gaussians to a compact .gtc file. Quantized parameters must already lie on their quantization grid.
    """
    quantized = gaussians.bits is not None
    flags = (FLAG_QUANTIZED * quantized + FLAG_INVERSE_SCALE * gaussians.inverse_scale +
             FLAG_TOPK_NORM * gaussians.topk_norm + FLAG_TILES * gaussians.tiles)
    bits = gaussians.bits if quantized else (32, 32, 32, 32)
    chunks = [HEADER.pack(MAGIC, gaussians.img_h, gaussians.img_w, len(gaussians.xy), flags, *bits, gaussians.gamma),
              struct.pack("<B", len(gaussians.input_channels))]
    for num_channels, fname in zip(gaussians.input_channels, gaussians.image_fnames):
        name = fname.encode("utf-8")
        chunks.append(struct.pack("<BB", num_channels, len(name)) + name)
    for param, num_bits in zip((gaussians.xy, gaussians.scale, gaussians.rot, gaussians.feat), bits):
        param = param.astype(np.float32).reshape(-1)
        if quantized:
            min_val, max_val = float(param.min()), float(param.max())
            codes, _ = quantize(param, num_bits, min_val, max_val)
            chunks.append(RANGE.pack(min_val, max_val) + pack_codes(codes, num_bits))
        else:
            chunks.append(param.astype("<f4").tobytes())
    with open(path, "wb") as file:
        file.write(b"".join(chunks))


def load_gtc(path):
    with open(path, "rb") as file:
        buffer = file.read()
    magic, img_h, img_w, num_gaussians, flags, *bits, gamma = HEADER.unpack_from(buffer, 0)
    if magic != MAGIC:
        raise ValueError(f"'{path}' is not a .gtc file")
    offset = HEADER.size
    num_images = buffer[offset]
    offset += 1
    input_channels, image_fnames = [], []
    for _ in range(num_images):
        num_channels, name_len = struct.unpack_from("<BB", buffer, offset)
        offset += 2
        input_channels.append(num_channels)
        image_fnames.append(buffer[offset:offset+name_len].decode("utf-8"))
        offset += name_len
    quantized = bool(flags & FLAG_QUANTIZED)
    params = []
    for dim, num_bits in zip((2, 2, 1, sum(input_channels)), bits):
        count = num_gaussians * dim
        if quantized:
            min_val, max_val = RANGE.unpack_from(buffer, offset)
            codes, offset = unpack_codes(buffer, offset+RANGE.size, count, num_bits)
            param = dequantize(codes, num_bits, min_val, max_val)
        else:
            param = np.frombuffer(buffer, dtype="<f4", count=count, offset=offset).astype(np.float32)
            offset += 4 * count
        params.append(param.reshape(num_gaussians, dim))
    return GaussianImage(*params, bits=tuple(bits) if quantized else None, img_h=img_h, img_w=img_w,
                         input_channels=input_channels, image_fnames=image_fnames, gamma=gamma,
                         inverse_scale=bool(flags & FLAG_INVERSE_SCALE), topk_norm=bool(flags & FLAG_TOPK_NORM),
                         tiles=bool(flags & FLAG_TILES))


def quantize(x, num_bits, min_val, max_val):
    """Same uniform quantization as utils.quantization_utils.ste_quantize()."""
    qmax = 2**num_bits - 1
    step = max((max_val - min_val) / qmax, 1e-8)
    codes = np.clip(np.round((x - min_val) / step), 0, qmax).astype(np.uint64)
    return codes, step


def dequantize(codes, num_bits, min_val, max_val):
    step = max((max_val - min_val) / (2**num_bits - 1), 1e-8)
    return (codes.astype(np.float32) * np.float32(step) + np.float32(min_val)).astype(np.float32)


def pack_codes(codes, num_bits):
    """Bit-pack unsigned integer codes, most significant bit first, padded to a whole byte."""
    shifts = np.arange(num_bits-1, -1, -1, dtype=np.uint64)
    bits = ((codes[:, None] >> shifts) & np.uint64(1)).astype(np.uint8)
    return np.packbits(bits.reshape(-1)).tobytes()


def unpack_codes(buffer, offset, count, num_bits):
    num_bytes = (count * num_bits + 7) // 8
    bits = np.unpackbits(np.frombuffer(buffer, dtype=np.uint8, count=num_bytes, offset=offset))[:count*num_bits]
    weights = np.left_shift(np.uint64(1), np.arange(num_bits-1, -1, -1, dtype=np.uint64))
    codes = (bits.reshape(count, num_bits).astype(np.uint64) * weights).sum(axis=1)
    return codes, offset + num_bytes

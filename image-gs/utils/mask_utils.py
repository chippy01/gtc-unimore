import functools
import os

import cv2
import numpy as np
import torch

EMPTY_MODES = ["none", "grey", "black", "white", "blurred", "uv"]
DETAIL_SIGMA = 1.0  # High-pass filter scale (pixels) for detecting blurred fill
DETAIL_WINDOW = 15  # Window size (pixels) for averaging local detail
MIN_EMPTY_AREA = 1e-4  # Empty regions smaller than this fraction of the image are kept (e.g. dark specks inside charts)


def compute_valid_masks(images, input_channels, image_fnames, load_path, mode, tol, dilate):
    """
    Detect texels that hold real content (True) as opposed to empty texture space (False).
    images: [C, H, W] array in [0, 1] (display space) holding one or more stacked images.
    Returns a [C, H, W] bool array where each image's mask is repeated over its channels.
    """
    if mode not in EMPTY_MODES:
        raise ValueError(f"Invalid empty_mode '{mode}', valid values are {EMPTY_MODES}")
    masks = []
    curr_channel = 0
    for num_channels, fname in zip(input_channels, image_fnames):
        image = images[curr_channel:curr_channel+num_channels].transpose(1, 2, 0)
        curr_channel += num_channels
        if mode == "uv":
            valid = get_uv_mask(load_path, fname, image.shape[:2])
        else:
            empty = detect_empty(image, mode, tol)
            empty = remove_small_regions(empty, min_area=MIN_EMPTY_AREA*empty.size)
            valid = ~empty
        # Keep a margin of empty texels around used areas so that filtering at chart borders stays correct
        if dilate > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate+1, 2*dilate+1))
            valid = cv2.dilate(valid.astype(np.uint8), kernel) > 0
        masks.append(np.repeat(valid[None], num_channels, axis=0))
    return np.concatenate(masks, axis=0)


def detect_empty(image, mode, tol):
    """
    image: [H, W, C] array in [0, 1]. tol is given in 8-bit levels.
    Flat fills (grey, black, white) match a single colour; blurred fills are found by their lack of fine detail.
    """
    image = image.astype(np.float32) * 255.0
    if mode == "blurred":
        gray = image.mean(axis=2)
        detail = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), DETAIL_SIGMA))
        detail = cv2.blur(detail, (DETAIL_WINDOW, DETAIL_WINDOW))
        return detail < tol
    if mode == "grey":
        fill = get_fill_grey(image)
    else:
        fill = 0.0 if mode == "black" else 255.0
    return np.all(np.abs(image - fill) <= tol, axis=2)


def get_fill_grey(image):
    """Most common neutral (R = G = B) level, since grey fills vary between tools (e.g. 127 vs. 128)."""
    neutral = (image.max(axis=2) - image.min(axis=2)) <= 1.0
    levels = np.round(image.mean(axis=2)[neutral]).astype(np.int64)
    return float(np.bincount(np.clip(levels, 0, 255), minlength=256).argmax())


def remove_small_regions(empty, min_area):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(empty.astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False  # Label 0 is the non-empty background
    return keep[labels]


def erode_mask(mask, radius):
    """mask: [H, W] bool array."""
    if radius <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    return cv2.erode(mask.astype(np.uint8), kernel) > 0


def masked_mean(values, mask=None):
    """Mean over the texels where mask is True (mask broadcasts against values), or a plain mean without a mask."""
    if mask is None:
        return values.mean()
    mask = mask.expand_as(values)
    return (values * mask).sum() / mask.sum().clamp(min=1)


def get_uv_mask(load_path, fname, shape):
    """Rasterize the UV charts of the .obj mesh stored next to the texture."""
    folder = load_path if os.path.isdir(load_path) else os.path.dirname(load_path)
    obj_files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".obj"))
    if len(obj_files) != 1:
        raise FileNotFoundError(f"empty_mode 'uv' needs exactly one .obj file in '{folder}' but found {len(obj_files)}")
    uvs, faces_per_mtl, mtl_files = load_obj_uvs(os.path.join(folder, obj_files[0]))
    tex_per_mtl = {}
    for mtl_file in mtl_files:
        if os.path.isfile(os.path.join(folder, mtl_file)):
            tex_per_mtl.update(load_mtl_textures(os.path.join(folder, mtl_file)))
    faces = [face for mtl, mtl_faces in faces_per_mtl.items()
             if os.path.splitext(os.path.basename(tex_per_mtl.get(mtl, "")))[0] == fname for face in mtl_faces]
    if len(faces) == 0 and len(faces_per_mtl) == 1:  # Single-material mesh without a usable texture reference
        faces = next(iter(faces_per_mtl.values()))
    if len(faces) == 0:
        raise ValueError(f"No face of '{obj_files[0]}' uses texture '{fname}'")
    height, width = shape
    shift = 4  # Sub-pixel precision bits for cv2.fillPoly
    mask = np.zeros(shape, dtype=np.uint8)
    for num_corners in sorted(set(len(face) for face in faces)):
        uv = uvs[np.asarray([face for face in faces if len(face) == num_corners])]  # [F, num_corners, 2]
        # Texel (i, j) has its center at UV ((j+0.5)/W, 1-(i+0.5)/H)
        xy = np.stack([uv[..., 0]*width - 0.5, (1.0-uv[..., 1])*height - 0.5], axis=-1)
        polygons = np.round(xy * (1 << shift)).astype(np.int32)
        cv2.fillPoly(mask, list(polygons), 1, lineType=cv2.LINE_8, shift=shift)
    return mask > 0


@functools.lru_cache(maxsize=1)
def load_obj_uvs(obj_path):
    """Returns UV coordinates [N, 2], {material: [face UV indices]}, and the referenced .mtl files."""
    uvs, faces_per_mtl, mtl_files = [], {}, []
    material = None
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            if line.startswith("vt "):
                tokens = line.split()
                uvs.append((float(tokens[1]), float(tokens[2])))
            elif line.startswith("f "):
                corners = [corner.split("/") for corner in line.split()[1:]]
                if all(len(corner) > 1 and corner[1] != "" for corner in corners):
                    # OBJ indices are 1-based, negative indices count back from the latest UV
                    face = [int(c[1])-1 if int(c[1]) > 0 else len(uvs)+int(c[1]) for c in corners]
                    faces_per_mtl.setdefault(material, []).append(face)
            elif line.startswith("usemtl "):
                material = line.split(maxsplit=1)[1].strip()
            elif line.startswith("mtllib "):
                mtl_files.append(line.split(maxsplit=1)[1].strip())
    return np.asarray(uvs, dtype=np.float64), faces_per_mtl, mtl_files


def load_mtl_textures(mtl_path):
    tex_per_mtl = {}
    material = None
    with open(mtl_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            tokens = line.split()
            if len(tokens) >= 2 and tokens[0] == "newmtl":
                material = tokens[1]
            elif len(tokens) >= 2 and tokens[0].lower() == "map_kd" and material is not None:
                tex_per_mtl[material] = tokens[-1]
    return tex_per_mtl

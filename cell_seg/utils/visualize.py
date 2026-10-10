"""Heatmaps and overlays. All functions return RGB uint8 H x W x 3."""
from typing import List

import cv2
import numpy as np
from matplotlib import colormaps
from scipy import ndimage
from skimage.segmentation import find_boundaries

from config import CellInfo

CONTOUR_RGB = (0, 255, 0)
SEED_RGB = (255, 0, 0)
SEED_RADIUS = 3
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _lut(name: str) -> np.ndarray:
    """256 x 3 uint8 RGB lookup table from a matplotlib colormap."""
    return (colormaps[name](np.linspace(0.0, 1.0, 256))[:, :3] * 255).round().astype(np.uint8)


_VIRIDIS = _lut("viridis")
_BONE = _lut("bone")


def prob_heatmap(prob: np.ndarray) -> np.ndarray:
    """Viridis on fixed [0, 1] scale (not min-max) so tau is comparable across images."""
    _check_2d(prob, "Probability map")
    idx = (np.clip(np.nan_to_num(prob.astype(np.float32)), 0.0, 1.0) * 255).round().astype(np.uint8)
    return _VIRIDIS[idx]


def mask_view(mask: np.ndarray) -> np.ndarray:
    """0/255 (or 0/1, bool) mask -> white-on-black RGB."""
    _check_2d(mask, "Mask")
    m = np.where(mask > 0, 255, 0).astype(np.uint8)
    return np.repeat(m[:, :, None], 3, axis=2)


def distance_map_view(dist: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Distance map with a red dot at each cell's seed."""
    _check_2d(dist, "Distance map")
    _check_labels(labels, dist.shape)
    peak = float(dist.max())
    idx = np.zeros(dist.shape, np.uint8) if peak <= 0 else (dist / peak * 255).round().astype(np.uint8)
    out = np.ascontiguousarray(_BONE[idx])
    ids = np.unique(labels[labels > 0])
    if len(ids):
        for r, c in ndimage.maximum_position(dist, labels, ids):
            cv2.circle(out, (int(c), int(r)), SEED_RADIUS, SEED_RGB, -1, lineType=cv2.LINE_AA)
    return out


def overlay_contours(img: np.ndarray, labels: np.ndarray, cells: List[CellInfo]) -> np.ndarray:
    """Green contour + white id (dark outline) per cell; labels must be renumbered (value == id)."""
    out = np.ascontiguousarray(_to_rgb_uint8(img))
    _check_labels(labels, out.shape[:2])
    if not cells:
        return out
    h, w = labels.shape
    keep = np.isin(labels, [c.id for c in cells])
    edges = find_boundaries(np.where(keep, labels, 0), mode="inner")
    thick = max(1, round(min(h, w) / 500))
    if thick > 1:
        edges = cv2.dilate(edges.astype(np.uint8), np.ones((thick, thick), np.uint8)) > 0
    out[edges] = CONTOUR_RGB
    scale = float(np.clip(min(h, w) / 700.0, 0.3, 1.2))
    t_fg = max(1, round(scale * 2))
    for c in cells:
        text = str(c.id)
        (tw, th), _ = cv2.getTextSize(text, FONT, scale, t_fg)
        org = (int(round(c.centroid[1] - tw / 2)), int(round(c.centroid[0] + th / 2)))
        cv2.putText(out, text, org, FONT, scale, (0, 0, 0), t_fg + 2, cv2.LINE_AA)
        cv2.putText(out, text, org, FONT, scale, (255, 255, 255), t_fg, cv2.LINE_AA)
    return out


def _to_rgb_uint8(img: np.ndarray) -> np.ndarray:
    if not isinstance(img, np.ndarray) or img.size == 0 or img.ndim not in (2, 3):
        raise ValueError(f"Image must be a non-empty H x W or H x W x C array, got "
                         f"{getattr(img, 'shape', type(img).__name__)}.")
    if img.dtype != np.uint8:
        f = np.nan_to_num(img.astype(np.float32))
        peak = float(f.max())
        scale = 255.0 if peak <= 1.0 else 255.0 / peak
        img = np.clip(f * scale, 0, 255).round().astype(np.uint8)
    if img.ndim == 2:
        return np.repeat(img[:, :, None], 3, axis=2)
    if img.shape[2] == 1:
        return np.repeat(img, 3, axis=2)
    if img.shape[2] in (3, 4):
        return img[:, :, :3].copy()
    raise ValueError(f"Image must have 1, 3 or 4 channels, got {img.shape[2]}.")


def _check_2d(a: np.ndarray, name: str) -> None:
    if not isinstance(a, np.ndarray) or a.ndim != 2 or a.size == 0:
        raise ValueError(f"{name} must be a non-empty 2-D array, got {getattr(a, 'shape', type(a).__name__)}.")


def _check_labels(labels: np.ndarray, shape) -> None:
    _check_2d(labels, "Labels")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"Labels must be integer, got {labels.dtype}.")
    if labels.shape != tuple(shape):
        raise ValueError(f"Labels shape {labels.shape} does not match image {tuple(shape)}.")

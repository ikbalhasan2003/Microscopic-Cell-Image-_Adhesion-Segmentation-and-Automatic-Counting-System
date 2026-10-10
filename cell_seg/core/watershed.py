"""Distance transform + watershed to split touching cells."""
from typing import Tuple

import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

CONNECTIVITY = np.ones((3, 3), bool)  # 8-connected, same as MaskRefiner


class WatershedSplitter:
    def run(self, img: np.ndarray, mask: np.ndarray, peak_ratio: float,
            min_dist: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (distance_map float32 H x W, labels int32 H x W; 0 = background, 1..N = cells)."""
        self._validate(img, mask, peak_ratio, min_dist)
        fg = mask > 0
        dist = cv2.distanceTransform(fg.astype(np.uint8), cv2.DIST_L2, 5)
        if not fg.any():
            return dist.astype(np.float32), np.zeros(fg.shape, np.int32)
        blobs, n_blobs = ndimage.label(fg, structure=CONNECTIVITY)
        markers, _ = ndimage.label(self._seeds(dist, blobs, n_blobs, peak_ratio, int(min_dist)),
                                   structure=CONNECTIVITY)
        labels = watershed(-dist, markers, mask=fg, connectivity=2)
        return dist.astype(np.float32), labels.astype(np.int32)

    @staticmethod
    def _seeds(dist: np.ndarray, blobs: np.ndarray, n_blobs: int,
               peak_ratio: float, min_dist: int) -> np.ndarray:
        """Boolean seed image: local maxima above peak_ratio * own-blob max; >= 1 seed per blob."""
        ids = np.arange(1, n_blobs + 1)
        blob_max = np.zeros(n_blobs + 1, np.float32)
        blob_max[1:] = ndimage.maximum(dist, blobs, ids)
        peaks = peak_local_max(dist, min_distance=min_dist, labels=blobs, exclude_border=False)
        seeds = np.zeros(dist.shape, bool)
        if len(peaks):
            r, c = peaks[:, 0], peaks[:, 1]
            keep = dist[r, c] >= peak_ratio * blob_max[blobs[r, c]]
            seeds[r[keep], c[keep]] = True
        missing = np.setdiff1d(ids, np.unique(blobs[seeds]))
        for r, c in ndimage.maximum_position(dist, blobs, missing):
            seeds[int(r), int(c)] = True
        return seeds

    @staticmethod
    def _validate(img: np.ndarray, mask: np.ndarray, peak_ratio: float, min_dist: int) -> None:
        for name, a in (("Image", img), ("Mask", mask)):
            if not isinstance(a, np.ndarray) or a.size == 0:
                raise ValueError(f"{name} is empty or not a numpy array.")
        if mask.ndim != 2:
            raise ValueError(f"Mask must be 2-D (H x W), got shape {mask.shape}.")
        if img.ndim not in (2, 3) or img.shape[:2] != mask.shape:
            raise ValueError(f"Image H x W {img.shape[:2]} does not match mask {mask.shape}.")
        if not 0.0 < peak_ratio < 1.0:
            raise ValueError(f"Peak ratio must be in (0, 1), got {peak_ratio}.")
        if int(min_dist) != min_dist or min_dist < 1:
            raise ValueError(f"Min peak distance must be an integer >= 1, got {min_dist}.")

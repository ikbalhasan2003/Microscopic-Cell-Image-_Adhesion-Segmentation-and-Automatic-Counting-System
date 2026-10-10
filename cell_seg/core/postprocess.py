"""Threshold + morphology clean-up."""
import cv2
import numpy as np
from scipy import ndimage

PROB_TOL = 1e-4                       # tolerated float error outside [0, 1]
CONNECTIVITY = np.ones((3, 3), bool)  # 8-connected components


class MaskRefiner:
    def run(self, prob: np.ndarray, thresh: float, kernel: int, min_area: int) -> np.ndarray:
        """H x W prob map in [0,1] -> H x W uint8 mask {0, 255}: threshold, open/close, fill holes, drop small blobs."""
        self._validate(prob, thresh, kernel, min_area)
        mask = (prob > thresh).astype(np.uint8)
        if kernel > 1:
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(kernel), int(kernel)))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se)
        filled = ndimage.binary_fill_holes(mask)
        return self._remove_small(filled, int(min_area)).astype(np.uint8) * 255

    @staticmethod
    def _validate(prob: np.ndarray, thresh: float, kernel: int, min_area: int) -> None:
        if not isinstance(prob, np.ndarray) or prob.size == 0:
            raise ValueError("Probability map is empty or not a numpy array.")
        if prob.ndim != 2:
            raise ValueError(f"Probability map must be 2-D (H x W), got shape {prob.shape}.")
        if not (np.issubdtype(prob.dtype, np.floating) or np.issubdtype(prob.dtype, np.integer)):
            raise ValueError(f"Probability map must be numeric, got dtype {prob.dtype}.")
        if not np.isfinite(prob).all():
            raise ValueError("Probability map contains NaN or inf.")
        lo, hi = float(prob.min()), float(prob.max())
        if lo < -PROB_TOL or hi > 1.0 + PROB_TOL:
            raise ValueError(f"Probability map values must be in [0, 1], got [{lo:.4f}, {hi:.4f}].")
        if not 0.0 < thresh < 1.0:
            raise ValueError(f"Threshold must be in (0, 1), got {thresh}.")
        if int(kernel) != kernel or kernel < 1:
            raise ValueError(f"Kernel size must be an integer >= 1, got {kernel}.")
        if int(min_area) != min_area or min_area < 0:
            raise ValueError(f"Min area must be an integer >= 0, got {min_area}.")

    @staticmethod
    def _remove_small(mask: np.ndarray, min_area: int) -> np.ndarray:
        """Drop connected components with fewer than min_area pixels."""
        if min_area <= 1 or not mask.any():
            return mask
        labels, _ = ndimage.label(mask, structure=CONNECTIVITY)
        areas = np.bincount(labels.ravel())
        keep = areas >= min_area
        keep[0] = False
        return keep[labels]

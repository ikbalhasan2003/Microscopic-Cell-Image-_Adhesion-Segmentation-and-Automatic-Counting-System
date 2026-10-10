"""Heatmaps and overlays. All functions return RGB uint8 H x W x 3."""
from typing import List

import numpy as np

from config import CellInfo


def prob_heatmap(prob: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def mask_view(mask: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def distance_map_view(dist: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Distance map with a red dot at each cell's seed."""
    raise NotImplementedError


def overlay_contours(img: np.ndarray, labels: np.ndarray, cells: List[CellInfo]) -> np.ndarray:
    raise NotImplementedError

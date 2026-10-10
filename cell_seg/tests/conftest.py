"""Shared fixtures: synthetic image with N known discs (two touching)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

H, W, R = 128, 160, 12
# 3 isolated discs + 1 touching pair (centres 22 px apart) -> 5 cells
CENTRES = [(30, 30), (30, 130), (100, 30), (95, 90), (95, 112)]
N_CELLS = len(CENTRES)


def _disc_mask() -> np.ndarray:
    yy, xx = np.mgrid[:H, :W]
    m = np.zeros((H, W), bool)
    for cy, cx in CENTRES:
        m |= (yy - cy) ** 2 + (xx - cx) ** 2 <= R ** 2
    return m


@pytest.fixture
def mask() -> np.ndarray:
    """uint8 {0, 255} mask of the discs."""
    return _disc_mask().astype(np.uint8) * 255


@pytest.fixture
def prob() -> np.ndarray:
    """float32 prob map: 0.9 inside discs, 0.05 outside."""
    return np.where(_disc_mask(), 0.9, 0.05).astype(np.float32)


@pytest.fixture
def rgb_image() -> np.ndarray:
    """RGB uint8: dark discs on light background with mild noise."""
    rng = np.random.default_rng(0)
    gray = np.where(_disc_mask(), 60, 200).astype(np.float32) + rng.normal(0, 5, (H, W))
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return np.stack([gray] * 3, axis=-1)

import numpy as np
import pytest
from scipy import ndimage

from core.postprocess import MaskRefiner
from tests.conftest import N_CELLS


def test_noise_removed_discs_kept(prob):
    noisy = prob.copy()
    rng = np.random.default_rng(0)
    noisy[(rng.random(prob.shape) < 0.01) & (prob < 0.5)] = 0.95
    m = MaskRefiner().run(noisy, 0.5, 3, 30)
    assert set(np.unique(m)) <= {0, 255}
    assert ndimage.label(m > 0)[1] == N_CELLS - 1  # touching pair = one blob


def test_hole_filled(prob):
    p = prob.copy()
    p[28:33, 28:33] = 0.0  # hole in first disc
    assert MaskRefiner().run(p, 0.5, 1, 0)[30, 30] == 255


def test_threshold_monotone(prob):
    smooth = ndimage.gaussian_filter(prob, 4)
    counts = [(MaskRefiner().run(smooth, t, 3, 10) > 0).sum() for t in (0.2, 0.4, 0.6, 0.8)]
    assert all(a >= b for a, b in zip(counts, counts[1:]))


def test_all_black_prob():
    assert MaskRefiner().run(np.zeros((50, 50), np.float32), 0.5, 5, 50).max() == 0


def test_single_pixel_below_min_area(prob):
    p = prob.copy()
    p[5, 80] = 0.99
    assert MaskRefiner().run(p, 0.5, 1, 2)[5, 80] == 0


@pytest.mark.parametrize("scale, thresh, kernel, min_area", [
    (1.5, 0.5, 5, 50), (1.0, 0.0, 5, 50), (1.0, 0.5, 0, 50), (1.0, 0.5, 5, -1)])
def test_invalid(prob, scale, thresh, kernel, min_area):
    with pytest.raises(ValueError):
        MaskRefiner().run(prob * scale, thresh, kernel, min_area)

import numpy as np
import pytest

from core.watershed import WatershedSplitter
from tests.conftest import N_CELLS


def n_labels(lab):
    return len(np.unique(lab)) - 1


def test_touching_pair_split(mask):
    dist, lab = WatershedSplitter().run(mask, mask, 0.4, 5)
    assert dist.dtype == np.float32 and lab.dtype == np.int32
    assert n_labels(lab) == N_CELLS
    assert not ((mask > 0) & (lab == 0)).any()


def test_empty_mask():
    z = np.zeros((40, 40), np.uint8)
    dist, lab = WatershedSplitter().run(z, z, 0.4, 5)
    assert dist.max() == 0 and lab.max() == 0


def test_min_dist_monotone(mask):
    counts = [n_labels(WatershedSplitter().run(mask, mask, 0.4, d)[1]) for d in (1, 5, 15, 40)]
    assert all(a >= b for a, b in zip(counts, counts[1:]))
    assert counts[-1] >= N_CELLS - 1  # every blob keeps >= 1 seed


@pytest.mark.parametrize("peak_ratio, min_dist", [(0.0, 5), (1.0, 5), (0.4, 0)])
def test_invalid(mask, peak_ratio, min_dist):
    with pytest.raises(ValueError):
        WatershedSplitter().run(mask, mask, peak_ratio, min_dist)


def test_shape_mismatch(mask):
    with pytest.raises(ValueError):
        WatershedSplitter().run(mask[:10], mask, 0.4, 5)

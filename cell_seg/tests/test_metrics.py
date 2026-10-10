import math

import numpy as np
import pytest

from utils import metrics


def test_iou_dice_identity_and_complement():
    x = np.random.default_rng(0).random((32, 32)) > 0.5
    assert metrics.iou(x, x) == 1.0 and metrics.dice(x, x) == 1.0
    assert metrics.iou(x, ~x) == 0.0


def test_known_example():
    p = np.zeros((4, 4), np.uint8)
    g = p.copy()
    p[0, :] = 1
    g[0, 2:] = 1
    g[1, 2:] = 1
    assert metrics.iou(p, g) == pytest.approx(2 / 6)
    assert metrics.dice(p, g) == pytest.approx(0.5)


def test_both_empty_is_perfect():
    z = np.zeros((8, 8), np.uint8)
    assert metrics.iou(z, z) == 1.0 and metrics.dice(z, z) == 1.0


def test_psnr_ssim(rgb_image):
    assert metrics.ssim(rgb_image, rgb_image) == pytest.approx(1.0)
    assert math.isinf(metrics.psnr(rgb_image, rgb_image))
    rng = np.random.default_rng(0)
    noisy = [np.clip(rgb_image + rng.normal(0, s, rgb_image.shape), 0, 255).astype(np.uint8)
             for s in (5, 20, 40)]
    vals = [metrics.psnr(rgb_image, n) for n in noisy]
    assert vals[0] > vals[1] > vals[2]


def test_count_error():
    assert metrics.count_error(9, 10) == pytest.approx(10.0)
    assert metrics.count_error(0, 0) == 0.0


def test_cell_precision_recall():
    lab = np.zeros((20, 20), np.int32)
    lab[2:8, 2:8] = 1
    lab[10:18, 10:18] = 2
    assert metrics.cell_precision_recall(lab, lab) == (1.0, 1.0)
    assert metrics.cell_precision_recall(np.where(lab == 2, 0, lab), lab) == (1.0, 0.5)


def test_shape_mismatch():
    with pytest.raises(ValueError):
        metrics.iou(np.zeros((4, 4)), np.zeros((5, 4)))

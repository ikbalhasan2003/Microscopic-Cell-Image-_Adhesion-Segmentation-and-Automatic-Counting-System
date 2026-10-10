import dataclasses

import numpy as np
import pytest
import torch

import core.pipeline as cp
from config import Params
from tests.conftest import N_CELLS

KEYS = {"original", "preprocessed", "prob", "mask", "distance", "overlay"}
PARAMS = Params(kernel=3, min_dist=5, min_area=30)


@pytest.fixture
def pipeline(monkeypatch, prob):
    """Pipeline with a fake U-Net that returns the synthetic prob map and counts calls."""
    calls = {"n": 0}

    class FakeSegmenter:
        def __init__(self, weights, device):
            pass

        def predict_prob(self, gray):
            calls["n"] += 1
            return prob

    monkeypatch.setattr(cp, "UNetSegmenter", FakeSegmenter)
    p = cp.CellCountingPipeline("unused.pth", torch.device("cpu"))
    p.calls = calls
    return p


def test_count_and_keys(pipeline, rgb_image):
    res = pipeline.run(rgb_image, PARAMS)
    assert res.count == N_CELLS == len(res.cells) == res.labels.max()
    assert set(res.images) == KEYS
    assert res.prob.dtype == np.float32 and set(np.unique(res.mask)) <= {0, 255}


def test_cache_hit_on_threshold_change(pipeline, rgb_image):
    pipeline.run(rgb_image, PARAMS)
    pipeline.run(rgb_image, dataclasses.replace(PARAMS, thresh=0.6, kernel=5))
    assert pipeline.calls["n"] == 1
    pipeline.run(rgb_image, dataclasses.replace(PARAMS, clip=3.0))
    assert pipeline.calls["n"] == 2


@pytest.mark.parametrize("bad", [dict(thresh=1.0), dict(kernel=0), dict(peak_ratio=0.0),
                                 dict(min_dist=0), dict(min_area=-5), dict(clip=0.0)])
def test_invalid_params(pipeline, rgb_image, bad):
    with pytest.raises(ValueError, match=list(bad)[0]):
        pipeline.run(rgb_image, dataclasses.replace(PARAMS, **bad))


def test_empty_image_rejected(pipeline):
    with pytest.raises(ValueError):
        pipeline.run(np.zeros((0, 0, 3), np.uint8), PARAMS)


def test_missing_weights_raised_on_run(rgb_image):
    p = cp.CellCountingPipeline("does/not/exist.pth", torch.device("cpu"))
    with pytest.raises(FileNotFoundError):
        p.run(rgb_image, PARAMS)

import numpy as np
import pytest

from core.preprocess import Preprocessor


@pytest.mark.parametrize("img", [
    np.zeros((64, 80, 3), np.uint8), np.zeros((64, 80), np.uint8), np.zeros((64, 80, 4), np.uint8),
    np.full((64, 80), 1000, np.uint16), np.random.default_rng(0).random((64, 80, 3)),
])
def test_shape_and_dtype(img):
    out = Preprocessor().run(img)
    assert out.shape == (64, 80) and out.dtype == np.uint8


def test_higher_clip_more_contrast():
    rng = np.random.default_rng(1)
    img = (120 + rng.normal(0, 4, (128, 128))).clip(0, 255).astype(np.uint8)
    p = Preprocessor()
    assert p.run(img, clip=4.0, blur_k=0).std() > p.run(img, clip=1.0, blur_k=0).std()


@pytest.mark.parametrize("k", [0, 1, 2, 4])
def test_any_blur_kernel(k):
    assert Preprocessor().run(np.zeros((40, 40), np.uint8), blur_k=k).shape == (40, 40)


@pytest.mark.parametrize("img", [np.array([]), np.zeros((10, 10), np.uint8),
                                 np.zeros((40, 40, 2), np.uint8), np.zeros((40, 40), np.int32)])
def test_invalid_input(img):
    with pytest.raises(ValueError):
        Preprocessor().run(img)

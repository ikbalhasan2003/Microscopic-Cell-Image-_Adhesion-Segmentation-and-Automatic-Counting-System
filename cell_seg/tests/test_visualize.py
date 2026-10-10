import numpy as np

from core.analysis import CellAnalyzer
from core.watershed import WatershedSplitter
from utils.visualize import distance_map_view, mask_view, overlay_contours, prob_heatmap


def test_all_views_rgb_uint8(rgb_image, prob, mask):
    dist, raw = WatershedSplitter().run(mask, mask, 0.4, 5)
    a = CellAnalyzer()
    lab, cells = a.renumber(raw, 10), a.run(raw, 10)
    h, w = mask.shape
    for out in (prob_heatmap(prob), mask_view(mask), distance_map_view(dist, lab),
                overlay_contours(rgb_image, lab, cells), overlay_contours(rgb_image[..., 0], lab, cells)):
        assert out.shape == (h, w, 3) and out.dtype == np.uint8


def test_heatmap_fixed_scale(prob):
    assert not np.array_equal(prob_heatmap(prob), prob_heatmap(prob * 0.5))


def test_overlay_without_cells_is_original(rgb_image):
    lab = np.zeros(rgb_image.shape[:2], np.int32)
    assert np.array_equal(overlay_contours(rgb_image, lab, []), rgb_image)

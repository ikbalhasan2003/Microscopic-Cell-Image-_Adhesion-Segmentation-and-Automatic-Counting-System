"""Orchestrator: runs all stages in order, returns every intermediate result."""
import hashlib
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from config import WEIGHTS_PATH, Params, PipelineResult
from core.analysis import CellAnalyzer
from core.postprocess import MaskRefiner
from core.preprocess import Preprocessor
from core.unet_model import UNetSegmenter
from core.watershed import WatershedSplitter
from utils.device import get_device
from utils.logger import get_logger, stage_timer
from utils.visualize import distance_map_view, mask_view, overlay_contours, prob_heatmap, to_rgb_uint8

MAX_SIDE = 4096  # larger images are downscaled (memory / UI response time)

_logger = get_logger(__name__)


class CellCountingPipeline:
    def __init__(self, weights: str = str(WEIGHTS_PATH),
                 device: Optional[torch.device] = None) -> None:
        """Create stage objects; the U-Net is loaded lazily on the first run."""
        self.weights = str(weights)
        self.device = device if device is not None else get_device()
        self.preprocessor = Preprocessor()
        self.refiner = MaskRefiner()
        self.splitter = WatershedSplitter()
        self.analyzer = CellAnalyzer()
        self._segmenter: Optional[UNetSegmenter] = None
        self._cache_key: Optional[Tuple] = None
        self._cache_prob: Optional[np.ndarray] = None

    def run(self, img: np.ndarray, params: Params) -> PipelineResult:
        """Preprocess -> U-Net (cached) -> refine -> watershed -> analyse -> RGB views."""
        self._validate_params(params)
        img, warnings = self._limit_size(img)
        with stage_timer(_logger, "preprocess"):
            gray = self.preprocessor.run(img, clip=params.clip, blur_k=params.blur_k)
            original = to_rgb_uint8(img)
        prob = self._prob(img, gray, params)
        with stage_timer(_logger, "refine"):
            mask = self.refiner.run(prob, params.thresh, params.kernel, params.min_area)
        with stage_timer(_logger, "watershed"):
            dist, raw_labels = self.splitter.run(gray, mask, params.peak_ratio, params.min_dist)
        with stage_timer(_logger, "analysis"):
            labels = self.analyzer.renumber(raw_labels, params.min_area)
            cells = self.analyzer.run(raw_labels, params.min_area)
        with stage_timer(_logger, "visualize"):
            images = {"original": original, "preprocessed": gray, "prob": prob_heatmap(prob),
                      "mask": mask_view(mask), "distance": distance_map_view(dist, labels),
                      "overlay": overlay_contours(original, labels, cells)}
        _logger.info("Detected %d cells", len(cells))
        return PipelineResult(images=images, cells=cells, count=len(cells),
                              prob=prob, mask=mask, labels=labels, warnings=warnings)

    @staticmethod
    def _limit_size(img: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Downscale so the longest side is <= MAX_SIDE; areas are then in downscaled pixels."""
        if not isinstance(img, np.ndarray) or img.ndim not in (2, 3) or img.size == 0:
            return img, []  # Preprocessor raises the proper ValueError
        h, w = img.shape[:2]
        if max(h, w) <= MAX_SIDE:
            return img, []
        scale = MAX_SIDE / max(h, w)
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        small = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        msg = (f"Image {w}x{h} downscaled by {scale:.3f} to {size[0]}x{size[1]}; "
               f"areas are in downscaled px (multiply by {1 / scale ** 2:.2f} for original px).")
        _logger.warning(msg)
        return small, [msg]

    def _prob(self, img: np.ndarray, gray: np.ndarray, params: Params) -> np.ndarray:
        """U-Net prob map; reused when image, clip and blur_k are unchanged (last entry only)."""
        key = (hashlib.sha1(np.ascontiguousarray(img).tobytes()).hexdigest(), img.shape,
               str(img.dtype), float(params.clip), int(params.blur_k))
        if key == self._cache_key and self._cache_prob is not None:
            _logger.info("unet skipped (cache hit)")
            return self._cache_prob
        with stage_timer(_logger, "unet"):
            if self._segmenter is None:
                self._segmenter = UNetSegmenter(self.weights, self.device)
            prob = self._segmenter.predict_prob(gray)
        self._cache_key, self._cache_prob = key, prob
        return prob

    @staticmethod
    def _validate_params(p: Params) -> None:
        if not isinstance(p, Params):
            raise ValueError(f"params must be a Params instance, got {type(p).__name__}.")
        checks = [
            ("thresh", 0.0 < p.thresh < 1.0, "in (0, 1)"),
            ("peak_ratio", 0.0 < p.peak_ratio < 1.0, "in (0, 1)"),
            ("clip", p.clip > 0, "> 0"),
            ("kernel", _is_int(p.kernel) and p.kernel >= 1, "an integer >= 1"),
            ("min_dist", _is_int(p.min_dist) and p.min_dist >= 1, "an integer >= 1"),
            ("min_area", _is_int(p.min_area) and p.min_area >= 0, "an integer >= 0"),
            ("blur_k", _is_int(p.blur_k) and p.blur_k >= 0, "an integer >= 0"),
        ]
        for name, ok, rule in checks:
            if not ok:
                raise ValueError(f"Params.{name} must be {rule}, got {getattr(p, name)!r}.")


def _is_int(v) -> bool:
    return isinstance(v, (int, np.integer, float)) and float(v).is_integer()

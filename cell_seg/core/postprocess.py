"""Threshold + morphology clean-up."""
import numpy as np


class MaskRefiner:
    def run(self, prob: np.ndarray, thresh: float, kernel: int, min_area: int) -> np.ndarray:
        raise NotImplementedError

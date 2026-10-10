"""Distance transform + watershed to split touching cells."""
from typing import Tuple

import numpy as np


class WatershedSplitter:
    def run(self, img: np.ndarray, mask: np.ndarray, peak_ratio: float,
            min_dist: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (distance_map, labels)."""
        raise NotImplementedError

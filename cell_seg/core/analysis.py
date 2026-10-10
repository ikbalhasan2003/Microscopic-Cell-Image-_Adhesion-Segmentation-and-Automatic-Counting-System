"""Cell numbering, area measurement, histogram."""
from typing import Dict, List

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from config import CellInfo


class CellAnalyzer:
    def renumber(self, labels: np.ndarray, min_area: int) -> np.ndarray:
        """Drop regions < min_area, renumber 1..N in reading order (top-left first)."""
        raise NotImplementedError

    def run(self, labels: np.ndarray, min_area: int) -> List[CellInfo]:
        raise NotImplementedError

    def histogram(self, cells: List[CellInfo]) -> Figure:
        raise NotImplementedError

    def summary(self, cells: List[CellInfo]) -> Dict[str, float]:
        """count, mean_area, median_area, std_area."""
        raise NotImplementedError

    def to_dataframe(self, cells: List[CellInfo]) -> pd.DataFrame:
        raise NotImplementedError

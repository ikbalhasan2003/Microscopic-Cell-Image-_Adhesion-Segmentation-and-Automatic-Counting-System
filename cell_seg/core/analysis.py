"""Cell numbering, area measurement, histogram."""
from typing import Dict, List

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from skimage.measure import regionprops

from config import CellInfo

CSV_COLUMNS = ["id", "area_px", "centroid_y", "centroid_x",
               "bbox_min_row", "bbox_min_col", "bbox_max_row", "bbox_max_col"]


class CellAnalyzer:
    def renumber(self, labels: np.ndarray, min_area: int) -> np.ndarray:
        """Drop regions < min_area, renumber 1..N in reading order (top-left first)."""
        self._validate(labels, min_area)
        regions = [r for r in regionprops(labels.astype(np.int32)) if r.area >= min_area]
        out = np.zeros(labels.shape, np.int32)
        if not regions:
            return out
        # Row bucket ~ typical cell height, so cells on one visual "line" read left to right.
        bucket = max(float(np.median([r.bbox[2] - r.bbox[0] for r in regions])), 1.0)
        regions.sort(key=lambda r: (int(r.centroid[0] // bucket), r.centroid[1]))
        lut = np.zeros(int(labels.max()) + 1, np.int32)
        for new_id, r in enumerate(regions, start=1):
            lut[r.label] = new_id
        return lut[labels]

    def run(self, labels: np.ndarray, min_area: int) -> List[CellInfo]:
        """Renumber, then measure id / area / centroid (row, col) / bbox per cell."""
        renumbered = self.renumber(labels, min_area)
        return [CellInfo(id=int(r.label), area=int(r.area),
                         centroid=(float(r.centroid[0]), float(r.centroid[1])),
                         bbox=tuple(int(v) for v in r.bbox))
                for r in regionprops(renumbered)]

    def histogram(self, cells: List[CellInfo]) -> Figure:
        """Cell area histogram with dashed mean line (matplotlib Figure, no pyplot)."""
        fig = Figure(figsize=(5, 3.5), dpi=100)
        ax = fig.add_subplot(111)
        if not cells:
            ax.text(0.5, 0.5, "No cells detected", ha="center", va="center",
                    fontsize=12, transform=ax.transAxes)
            ax.set_axis_off()
            return fig
        areas = np.array([c.area for c in cells])
        ax.hist(areas, bins="auto", color="#1D45D8", edgecolor="white")
        mean, median = float(areas.mean()), float(np.median(areas))
        ax.axvline(mean, color="#0F766E", linestyle="--", label=f"mean = {mean:.0f}")
        ax.axvline(median, color="#64748B", linestyle=":", label=f"median = {median:.0f}")
        ax.set_xlabel("Cell area (px²)")
        ax.set_ylabel("Count")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False)
        fig.tight_layout()
        return fig

    def summary(self, cells: List[CellInfo]) -> Dict[str, float]:
        """count, mean_area, median_area, std_area."""
        if not cells:
            return {"count": 0, "mean_area": 0.0, "median_area": 0.0, "std_area": 0.0}
        areas = np.array([c.area for c in cells], np.float64)
        return {"count": len(cells), "mean_area": float(areas.mean()),
                "median_area": float(np.median(areas)), "std_area": float(areas.std())}

    def to_dataframe(self, cells: List[CellInfo]) -> pd.DataFrame:
        """One row per cell, columns CSV_COLUMNS."""
        rows = [(c.id, c.area, round(c.centroid[0], 2), round(c.centroid[1], 2), *c.bbox)
                for c in cells]
        return pd.DataFrame(rows, columns=CSV_COLUMNS)

    @staticmethod
    def _validate(labels: np.ndarray, min_area: int) -> None:
        if not isinstance(labels, np.ndarray) or labels.ndim != 2 or labels.size == 0:
            raise ValueError(f"Labels must be a non-empty 2-D array, got "
                             f"{getattr(labels, 'shape', type(labels).__name__)}.")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f"Labels must be an integer array, got {labels.dtype}.")
        if labels.min() < 0:
            raise ValueError("Labels must be non-negative (0 = background).")
        if int(min_area) != min_area or min_area < 0:
            raise ValueError(f"Min area must be an integer >= 0, got {min_area}.")

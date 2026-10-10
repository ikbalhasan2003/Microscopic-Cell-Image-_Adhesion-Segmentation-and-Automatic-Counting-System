"""Default parameters, paths and shared @dataclass types."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT_DIR / "weights" / "unet_cells.pth"
SAMPLES_DIR = ROOT_DIR / "samples"
OUTPUTS_DIR = ROOT_DIR / "outputs"


@dataclass
class Params:
    """All slider values from the UI."""
    clip: float = 2.0          # CLAHE clip limit
    blur_k: int = 3            # Gaussian blur kernel
    thresh: float = 0.5        # probability threshold tau
    kernel: int = 5            # morphology kernel size
    peak_ratio: float = 0.4    # distance-map peak ratio r
    min_dist: int = 10         # min peak distance (px)
    min_area: int = 50         # min cell area (px^2)


@dataclass
class CellInfo:
    id: int
    area: int
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]


@dataclass
class PipelineResult:
    images: Dict[str, np.ndarray] = field(default_factory=dict)  # RGB uint8 views for the UI
    cells: List[CellInfo] = field(default_factory=list)
    count: int = 0
    prob: Optional[np.ndarray] = None    # raw H x W probability map, float32 [0, 1]
    mask: Optional[np.ndarray] = None    # raw binary mask, uint8 {0, 255}
    labels: Optional[np.ndarray] = None  # renumbered label image, int32, ids match cells

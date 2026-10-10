"""Orchestrator: runs all stages in order, returns every intermediate result."""
from typing import Optional

import numpy as np
import torch

from config import WEIGHTS_PATH, Params, PipelineResult


class CellCountingPipeline:
    def __init__(self, weights: str = str(WEIGHTS_PATH),
                 device: Optional[torch.device] = None) -> None:
        raise NotImplementedError

    def run(self, img: np.ndarray, params: Params) -> PipelineResult:
        raise NotImplementedError

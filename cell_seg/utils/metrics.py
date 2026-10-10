"""IoU, Dice, PSNR, SSIM, per-cell precision / recall, count error."""
from typing import Tuple

import numpy as np


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    raise NotImplementedError


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    raise NotImplementedError


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    raise NotImplementedError


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    raise NotImplementedError


def count_error(pred_count: int, gt_count: int) -> float:
    """|pred - gt| / gt * 100."""
    raise NotImplementedError


def cell_precision_recall(pred_labels: np.ndarray, gt_labels: np.ndarray,
                          iou_thr: float = 0.5) -> Tuple[float, float]:
    raise NotImplementedError

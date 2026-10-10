"""IoU, Dice, PSNR, SSIM, per-cell precision / recall, count error."""
from typing import Tuple

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

SSIM_MIN_SIDE = 7  # skimage default window size


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """|P ∩ G| / |P ∪ G| on binarised (> 0) masks; both empty -> 1.0."""
    p, g = _binary_pair(pred, gt)
    union = np.logical_or(p, g).sum()
    return 1.0 if union == 0 else float(np.logical_and(p, g).sum() / union)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """2|P ∩ G| / (|P| + |G|) on binarised (> 0) masks; both empty -> 1.0."""
    p, g = _binary_pair(pred, gt)
    total = p.sum() + g.sum()
    return 1.0 if total == 0 else float(2.0 * np.logical_and(p, g).sum() / total)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR in dB, data_range from dtype; identical images -> inf."""
    rng = _image_pair(a, b)
    if np.array_equal(a, b):
        return float("inf")
    return float(peak_signal_noise_ratio(a, b, data_range=rng))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM, data_range from dtype; RGB/RGBA handled with channel_axis=-1."""
    rng = _image_pair(a, b)
    if min(a.shape[:2]) < SSIM_MIN_SIDE:
        raise ValueError(f"SSIM needs images at least {SSIM_MIN_SIDE}x{SSIM_MIN_SIDE}, got {a.shape[:2]}.")
    channel_axis = -1 if a.ndim == 3 else None
    return float(structural_similarity(a, b, data_range=rng, channel_axis=channel_axis))


def count_error(pred_count: int, gt_count: int) -> float:
    """|pred - gt| / gt * 100; gt = 0 uses denominator 1 (so 0 vs 0 -> 0 %)."""
    if int(pred_count) != pred_count or int(gt_count) != gt_count or pred_count < 0 or gt_count < 0:
        raise ValueError(f"Counts must be integers >= 0, got pred={pred_count}, gt={gt_count}.")
    return abs(int(pred_count) - int(gt_count)) / max(int(gt_count), 1) * 100.0


def cell_precision_recall(pred_labels: np.ndarray, gt_labels: np.ndarray,
                          iou_thr: float = 0.5) -> Tuple[float, float]:
    """Greedy one-to-one instance matching by IoU >= iou_thr -> (precision, recall)."""
    if not isinstance(pred_labels, np.ndarray) or not isinstance(gt_labels, np.ndarray):
        raise ValueError("Label images must be numpy arrays.")
    if pred_labels.shape != gt_labels.shape or pred_labels.ndim != 2:
        raise ValueError(f"Label images must be 2-D with equal shape, got {pred_labels.shape} and {gt_labels.shape}.")
    if not (np.issubdtype(pred_labels.dtype, np.integer) and np.issubdtype(gt_labels.dtype, np.integer)):
        raise ValueError("Label images must be integer arrays.")
    if not 0.0 < iou_thr <= 1.0:
        raise ValueError(f"iou_thr must be in (0, 1], got {iou_thr}.")
    p_ids, p_area = np.unique(pred_labels[pred_labels > 0], return_counts=True)
    g_ids, g_area = np.unique(gt_labels[gt_labels > 0], return_counts=True)
    if len(p_ids) == 0 and len(g_ids) == 0:
        return 1.0, 1.0
    if len(p_ids) == 0 or len(g_ids) == 0:
        return 0.0, 0.0
    both = (pred_labels > 0) & (gt_labels > 0)
    pairs, inter = np.unique(np.stack([pred_labels[both], gt_labels[both]]), axis=1, return_counts=True)
    p_area_of = dict(zip(p_ids.tolist(), p_area.tolist()))
    g_area_of = dict(zip(g_ids.tolist(), g_area.tolist()))
    candidates = []
    for (pi, gi), n in zip(pairs.T.tolist(), inter.tolist()):
        score = n / (p_area_of[pi] + g_area_of[gi] - n)
        if score >= iou_thr:
            candidates.append((score, pi, gi))
    used_p, used_g = set(), set()
    for _, pi, gi in sorted(candidates, reverse=True):
        if pi not in used_p and gi not in used_g:
            used_p.add(pi)
            used_g.add(gi)
    matched = len(used_p)
    return matched / len(p_ids), matched / len(g_ids)


def _binary_pair(pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(pred, np.ndarray) or not isinstance(gt, np.ndarray):
        raise ValueError("Masks must be numpy arrays.")
    if pred.shape != gt.shape:
        raise ValueError(f"Mask shapes differ: {pred.shape} vs {gt.shape}.")
    return pred > 0, gt > 0


def _image_pair(a: np.ndarray, b: np.ndarray) -> float:
    """Validate two images; return data_range (255 uint8, 65535 uint16, 1.0 float)."""
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray) or a.size == 0:
        raise ValueError("Images must be non-empty numpy arrays.")
    if a.shape != b.shape:
        raise ValueError(f"Image shapes differ: {a.shape} vs {b.shape}.")
    if a.dtype != b.dtype:
        raise ValueError(f"Image dtypes differ: {a.dtype} vs {b.dtype}.")
    if a.ndim not in (2, 3):
        raise ValueError(f"Images must be H x W or H x W x C, got {a.shape}.")
    if a.dtype == np.uint8:
        return 255.0
    if a.dtype == np.uint16:
        return 65535.0
    if np.issubdtype(a.dtype, np.floating):
        return 1.0
    raise ValueError(f"Unsupported image dtype {a.dtype}; use uint8, uint16 or float in [0, 1].")

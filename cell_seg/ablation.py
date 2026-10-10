"""Ablation: same image, 3 parameter sets -> count, PSNR, SSIM, runtime table."""
import time
from typing import Dict

import cv2
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from skimage.segmentation import find_boundaries

from config import OUTPUTS_DIR, SAMPLES_DIR, Params
from core.pipeline import CellCountingPipeline
from utils import metrics
from utils.logger import get_logger

SAMPLE = "02_dense_touching"  # touching cells + ground truth available
# A-C vary τ / kernel / r around the original mockup defaults (min_dist 10, min_area 50);
# D = current Params() defaults, chosen by tuning.py on the val split.
SETS: Dict[str, Params] = {
    "A: lenient τ0.3 k3 r0.3": Params(thresh=0.3, kernel=3, peak_ratio=0.3, min_dist=10, min_area=50),
    "B: mockup τ0.5 k5 r0.4": Params(thresh=0.5, kernel=5, peak_ratio=0.4, min_dist=10, min_area=50),
    "C: strict τ0.7 k7 r0.6": Params(thresh=0.7, kernel=7, peak_ratio=0.6, min_dist=10, min_area=50),
    "D: tuned k3 a20 (new defaults)": Params(),
}

_logger = get_logger("ablation")


def run_ablation() -> None:
    img = cv2.cvtColor(cv2.imread(str(SAMPLES_DIR / f"{SAMPLE}.png")), cv2.COLOR_BGR2RGB)
    gt_labels = cv2.imread(str(SAMPLES_DIR / "gt" / f"{SAMPLE}_labels.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
    gt_mask = np.where(gt_labels > 0, 255, 0).astype(np.uint8)
    pipeline = CellCountingPipeline()
    t0 = time.perf_counter()
    pipeline.run(img, Params())  # warm-up: loads U-Net, fills prob cache (clip/blur equal in all sets)
    unet_s = time.perf_counter() - t0

    rows, overlays = [], {}
    for name, p in SETS.items():
        t = time.perf_counter()
        res = pipeline.run(img, p)
        runtime = time.perf_counter() - t
        pr, rc = metrics.cell_precision_recall(res.labels, gt_labels)
        rows.append({"set": name, "thresh": p.thresh, "kernel": p.kernel, "peak_ratio": p.peak_ratio,
                     "min_dist": p.min_dist, "min_area": p.min_area, "count": res.count,
                     "gt_count": int(gt_labels.max()),
                     "mean_area": pipeline.analyzer.summary(res.cells)["mean_area"],
                     "iou": metrics.iou(res.mask, gt_mask), "dice": metrics.dice(res.mask, gt_mask),
                     "psnr_db": metrics.psnr(res.mask, gt_mask), "ssim": metrics.ssim(res.mask, gt_mask),
                     "cell_precision": pr, "cell_recall": rc, "runtime_s": runtime})
        overlays[name] = res.images["overlay"]

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUTS_DIR / "ablation.csv", index=False, float_format="%.4f")
    _save_figure(img, gt_labels, overlays, df)
    _logger.info("Ablation on %s (U-Net + load %.2f s once, runtime = post-processing with cached prob):\n%s",
                 SAMPLE, unet_s, df.drop(columns=["thresh", "kernel", "peak_ratio", "min_dist", "min_area"])
                 .to_string(index=False, float_format="%.3f"))


def _save_figure(img, gt_labels, overlays, df) -> None:
    cols = len(overlays) + 1
    fig = Figure(figsize=(4 * cols, 4.2))
    ax = fig.add_subplot(1, cols, 1)
    ax.imshow(_gt_view(img, gt_labels))
    ax.set_title(f"Ground truth ({int(gt_labels.max())} cells)", fontsize=10)
    ax.axis("off")
    for i, (name, ov) in enumerate(overlays.items(), start=2):
        r = df[df["set"] == name].iloc[0]
        ax = fig.add_subplot(1, cols, i)
        ax.imshow(ov)
        ax.set_title(f"{name}\ncount {r['count']} | Dice {r['dice']:.3f} | SSIM {r['ssim']:.3f}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(OUTPUTS_DIR / "ablation.png"), dpi=130)


def _gt_view(img: np.ndarray, gt_labels: np.ndarray) -> np.ndarray:
    """Yellow GT instance boundaries on the image."""
    out = img.copy()
    out[find_boundaries(gt_labels, mode="inner")] = (255, 255, 0)
    return out


if __name__ == "__main__":
    run_ablation()

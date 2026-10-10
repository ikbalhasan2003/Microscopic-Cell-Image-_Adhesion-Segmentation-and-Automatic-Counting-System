"""Grid search of post-processing parameters on the val split; compare default vs best on test.

Usage (from cell_seg/): python tuning.py --data D:/datasets/dsb2018/stage1_train
U-Net runs once per image; only MaskRefiner / WatershedSplitter / CellAnalyzer are re-run per setting.
"""
import argparse
import dataclasses
import itertools
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import OUTPUTS_DIR, WEIGHTS_PATH, Params
from core.analysis import CellAnalyzer
from core.postprocess import MaskRefiner
from core.preprocess import Preprocessor
from core.unet_model import UNetSegmenter
from core.watershed import WatershedSplitter
from training.dataset import read_instances, read_rgb, split_ids
from utils import metrics
from utils.device import get_device
from utils.logger import get_logger

GRID = {
    "thresh": [0.4, 0.5, 0.6],
    "kernel": [3, 5],
    "peak_ratio": [0.2, 0.3, 0.4, 0.5],
    "min_dist": [3, 5, 7, 10],
    "min_area": [10, 20, 50],
}

_logger = get_logger("tuning")
Sample = Tuple[np.ndarray, np.ndarray, np.ndarray]  # gray, prob, gt_labels


def load_split(data: str, split: str) -> List[Sample]:
    """Preprocess + U-Net once per image of the split (native resolution)."""
    seg, pre, ids = UNetSegmenter(str(WEIGHTS_PATH), get_device()), Preprocessor(), split_ids(data)
    out = []
    for sid in ids[split]:
        sample = Path(data) / sid
        gray = pre.run(read_rgb(sample / "images" / f"{sid}.png"))
        out.append((gray, seg.predict_prob(gray), read_instances(sample)))
    return out


def evaluate(samples: List[Sample], p: Params) -> Dict[str, float]:
    """Mean per-cell P / R / F1, count error and mask Dice over the samples."""
    ref, ws, ana = MaskRefiner(), WatershedSplitter(), CellAnalyzer()
    prec, rec, err, dsc = [], [], [], []
    for gray, prob, gt in samples:
        mask = ref.run(prob, p.thresh, p.kernel, p.min_area)
        _, raw = ws.run(gray, mask, p.peak_ratio, p.min_dist)
        labels = ana.renumber(raw, p.min_area)
        pr, rc = metrics.cell_precision_recall(labels, gt)
        prec.append(pr)
        rec.append(rc)
        err.append(metrics.count_error(int(labels.max()), int(len(np.unique(gt)) - 1)))
        dsc.append(metrics.dice(mask, gt))
    p_m, r_m = float(np.mean(prec)), float(np.mean(rec))
    return {"precision": p_m, "recall": r_m, "f1": 2 * p_m * r_m / max(p_m + r_m, 1e-9),
            "count_err_median": float(np.median(err)), "count_err_mean": float(np.mean(err)),
            "dice": float(np.mean(dsc))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="path to stage1_train/")
    args = ap.parse_args()

    t0 = time.perf_counter()
    val = load_split(args.data, "val")
    _logger.info("val split loaded (%d images) in %.1f s", len(val), time.perf_counter() - t0)
    rows = []
    keys = list(GRID)
    for values in itertools.product(*GRID.values()):
        p = dataclasses.replace(Params(), **dict(zip(keys, values)))
        rows.append({**dict(zip(keys, values)), **evaluate(val, p)})
    df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    df.to_csv(OUTPUTS_DIR / "tuning_val.csv", index=False)
    best = df.iloc[0]
    best_params = dataclasses.replace(Params(), **{k: type(getattr(Params(), k))(best[k]) for k in keys})
    _logger.info("%d settings searched in %.1f s; top 5 on val:\n%s", len(df), time.perf_counter() - t0,
                 df.head(5).to_string(index=False, float_format="%.4f"))

    test = load_split(args.data, "test")
    result = {"default_params": dataclasses.asdict(Params()), "tuned_params": dataclasses.asdict(best_params),
              "val_default": evaluate(val, Params()), "val_tuned": evaluate(val, best_params),
              "test_default": evaluate(test, Params()), "test_tuned": evaluate(test, best_params)}
    (OUTPUTS_DIR / "tuning_test.json").write_text(json.dumps(result, indent=2))
    table = pd.DataFrame({k: v for k, v in result.items() if k.startswith(("val_", "test_"))}).T
    _logger.info("default vs tuned:\n%s", table.to_string(float_format="%.4f"))


if __name__ == "__main__":
    main()

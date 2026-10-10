"""Fine-tuning loop: Adam lr=1e-4, batch 8, 50 epochs, early stop on val Dice."""
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from matplotlib.figure import Figure
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python training/train.py`

from config import OUTPUTS_DIR, WEIGHTS_PATH, Params  # noqa: E402
from core.analysis import CellAnalyzer  # noqa: E402
from core.postprocess import MaskRefiner  # noqa: E402
from core.preprocess import Preprocessor  # noqa: E402
from core.unet_model import UNetSegmenter, build_unet  # noqa: E402
from core.watershed import WatershedSplitter  # noqa: E402
from training.dataset import CLIP, BLUR_K, DSB2018Dataset, read_instances, read_rgb  # noqa: E402
from training.losses import BCEDiceLoss  # noqa: E402
from utils.device import get_device  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from utils import metrics  # noqa: E402

SEED = 42
HISTORY_CSV = OUTPUTS_DIR / "train_history.csv"
CURVES_PNG = OUTPUTS_DIR / "train_curves.png"
TEST_JSON = OUTPUTS_DIR / "test_metrics.json"

_logger = get_logger("train")


def parse_args() -> argparse.Namespace:
    """--data --epochs --batch --lr --patience --size --workers --limit --out"""
    p = argparse.ArgumentParser(description="Fine-tune ResNet34 U-Net on DSB 2018 stage1_train.")
    p.add_argument("--data", required=True, help="path to stage1_train/")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8, help="early stop after N epochs without val Dice gain")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--workers", type=int, default=2, help="DataLoader workers (use 0 if Windows has issues)")
    p.add_argument("--limit", type=int, default=0, help="use only N train images (val/test N//4) for smoke runs")
    p.add_argument("--out", default=str(WEIGHTS_PATH), help="checkpoint path")
    args = p.parse_args()
    for name in ("epochs", "batch", "patience", "size"):
        if getattr(args, name) < 1:
            p.error(f"--{name} must be >= 1")
    if args.lr <= 0 or args.workers < 0 or args.limit < 0:
        p.error("--lr must be > 0, --workers and --limit must be >= 0")
    return args


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _limit(ds: DSB2018Dataset, n: int):
    return Subset(ds, range(min(n, len(ds)))) if n else ds


def _run_epoch(model, loader, loss_fn, device, optimizer=None, scaler=None) -> Dict[str, float]:
    """One pass; trains if optimizer given. Returns loss (+ mean per-image IoU/Dice when evaluating)."""
    train = optimizer is not None
    model.train(train)
    total, n, ious, dices = 0.0, 0, [], []
    use_amp = scaler is not None
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = loss_fn(logits.float(), y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            else:
                pred = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                gt = y.cpu().numpy() > 0.5
                ious += [metrics.iou(p[0], g[0]) for p, g in zip(pred, gt)]
                dices += [metrics.dice(p[0], g[0]) for p, g in zip(pred, gt)]
            total += loss.item() * x.size(0)
            n += x.size(0)
    out = {"loss": total / max(n, 1)}
    if not train:
        out.update(iou=float(np.mean(ious)), dice=float(np.mean(dices)))
    return out


def _save_history(history: List[Dict[str, float]]) -> None:
    with open(HISTORY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    ep = [h["epoch"] for h in history]
    fig = Figure(figsize=(10, 4))
    ax = fig.add_subplot(121)
    ax.plot(ep, [h["train_loss"] for h in history], label="train loss")
    ax.plot(ep, [h["val_loss"] for h in history], label="val loss")
    ax.set_xlabel("epoch"), ax.set_ylabel("BCE + Dice loss"), ax.legend(), ax.set_title("Loss")
    ax = fig.add_subplot(122)
    ax.plot(ep, [h["val_dice"] for h in history], label="val Dice")
    ax.plot(ep, [h["val_iou"] for h in history], label="val IoU")
    ax.set_xlabel("epoch"), ax.set_ylim(0, 1), ax.legend(), ax.set_title("Validation")
    fig.tight_layout()
    fig.savefig(str(CURVES_PNG), dpi=120)


def _evaluate_test(ckpt: Path, test_ds: DSB2018Dataset, n: int, device: torch.device,
                   loss_fn, batch: int) -> Dict[str, float]:
    """Best checkpoint on test split: 256 px metrics + full-pipeline metrics at native resolution."""
    model = build_unet().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True)["model"])
    subset = _limit(test_ds, n)
    res_256 = _run_epoch(model, DataLoader(subset, batch_size=batch), loss_fn, device)

    seg, pre, ref, ws, ana, prm = (UNetSegmenter(str(ckpt), device), Preprocessor(), MaskRefiner(),
                                   WatershedSplitter(), CellAnalyzer(), Params())
    ious, dices, errs, precs, recs = [], [], [], [], []
    for sid in test_ds.ids[:len(subset)]:
        sample = test_ds.root / sid
        gt_labels = read_instances(sample)
        gray = pre.run(read_rgb(sample / "images" / f"{sid}.png"), clip=CLIP, blur_k=BLUR_K)
        prob = seg.predict_prob(gray)
        mask = ref.run(prob, prm.thresh, prm.kernel, prm.min_area)
        _, raw = ws.run(gray, mask, prm.peak_ratio, prm.min_dist)
        labels = ana.renumber(raw, prm.min_area)
        ious.append(metrics.iou(mask, gt_labels))
        dices.append(metrics.dice(mask, gt_labels))
        errs.append(metrics.count_error(int(labels.max()), int(len(np.unique(gt_labels)) - 1)))
        p, r = metrics.cell_precision_recall(labels, gt_labels)
        precs.append(p)
        recs.append(r)
    return {"test_images": len(subset), "test_loss_256": res_256["loss"],
            "test_iou_256": res_256["iou"], "test_dice_256": res_256["dice"],
            "native_mask_iou": float(np.mean(ious)), "native_mask_dice": float(np.mean(dices)),
            "count_error_pct_mean": float(np.mean(errs)), "count_error_pct_median": float(np.median(errs)),
            "cell_precision@0.5": float(np.mean(precs)), "cell_recall@0.5": float(np.mean(recs))}


def train() -> None:
    args = parse_args()
    _seed_everything(SEED)
    device = get_device()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    train_ds = DSB2018Dataset(args.data, "train", args.size)
    val_ds = DSB2018Dataset(args.data, "val", args.size)
    test_ds = DSB2018Dataset(args.data, "test", args.size)
    n_eval = max(args.limit // 4, 2) if args.limit else 0
    g = torch.Generator().manual_seed(SEED)
    train_dl = DataLoader(_limit(train_ds, args.limit), batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, generator=g, drop_last=False,
                          pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    val_dl = DataLoader(_limit(val_ds, n_eval), batch_size=args.batch, num_workers=0)

    model = build_unet(encoder_weights="imagenet").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = BCEDiceLoss()
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    _logger.info("Training: %d train / %d val images, %d epochs max, batch %d, lr %g, AMP %s",
                 len(train_dl.dataset), len(val_dl.dataset), args.epochs, args.batch, args.lr,
                 scaler is not None)

    history: List[Dict[str, float]] = []
    best_dice, best_epoch, stale = -1.0, 0, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        tr = _run_epoch(model, train_dl, loss_fn, device, optimizer, scaler)
        va = _run_epoch(model, val_dl, loss_fn, device)
        history.append({"epoch": epoch, "train_loss": tr["loss"], "val_loss": va["loss"],
                        "val_iou": va["iou"], "val_dice": va["dice"], "time_s": time.perf_counter() - t0})
        improved = va["dice"] > best_dice
        if improved:
            best_dice, best_epoch, stale = va["dice"], epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_dice": best_dice}, out)
        else:
            stale += 1
        _logger.info("epoch %3d | train %.4f | val %.4f | IoU %.4f | Dice %.4f | %.1f s%s",
                     epoch, tr["loss"], va["loss"], va["iou"], va["dice"], history[-1]["time_s"],
                     " | saved" if improved else "")
        _save_history(history)
        if stale >= args.patience:
            _logger.info("Early stop: no val Dice gain for %d epochs", args.patience)
            break

    if best_epoch == 0:  # e.g. NaN val Dice: never evaluate a stale checkpoint from an older run
        raise RuntimeError(f"No checkpoint saved in this run (val Dice never valid); {out} not updated.")
    _logger.info("Best val Dice %.4f at epoch %d -> %s", best_dice, best_epoch, out)
    results = {"best_epoch": best_epoch, "best_val_dice": best_dice,
               **_evaluate_test(out, test_ds, n_eval, device, loss_fn, args.batch)}
    TEST_JSON.write_text(json.dumps(results, indent=2))
    width = max(len(k) for k in results)
    table = "\n".join(f"  {k:<{width}}  {v:.4f}" if isinstance(v, float) else f"  {k:<{width}}  {v}"
                      for k, v in results.items())
    _logger.info("Test results (%s):\n%s", TEST_JSON, table)


if __name__ == "__main__":
    train()

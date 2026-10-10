"""U-Net (ResNet34 encoder) wrapper: per-pixel cell probability map."""
from pathlib import Path
from typing import Dict, List

import numpy as np
import segmentation_models_pytorch as smp
import torch

from utils.logger import get_logger

# ImageNet normalisation; training/dataset.py imports these so train and inference match.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
DIVISOR = 32  # ResNet34 encoder downsamples 5x
TILE, OVERLAP = 512, 64  # tiled fallback after CUDA OOM

_logger = get_logger(__name__)


def build_unet(encoder_weights=None) -> torch.nn.Module:
    """ResNet34 U-Net with 1 output channel (logits); shared with training/train.py."""
    return smp.Unet(encoder_name="resnet34", encoder_weights=encoder_weights,
                    in_channels=3, classes=1)


class UNetSegmenter:
    def __init__(self, weights: str, device: torch.device) -> None:
        path = Path(weights)
        if not path.is_file():
            raise FileNotFoundError(
                f"U-Net weights not found at {path}. Train with training/train.py or copy "
                "unet_cells.pth into cell_seg/weights/.")
        self.model = build_unet()
        try:
            self.model.load_state_dict(self._read_state_dict(path, device))
        except RuntimeError as e:
            raise ValueError(f"Weights file {path.name} does not match the ResNet34 U-Net "
                             "(wrong architecture or corrupted file).") from e
        self.model.eval()
        self._to_device(device)
        _logger.info("U-Net loaded from %s on %s", path, device)

    def _to_device(self, device: torch.device) -> None:
        self.device = device
        self.model.to(device)
        self._mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    @staticmethod
    def _read_state_dict(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
        """Accept a raw state_dict or a checkpoint dict with key 'model'."""
        ckpt = torch.load(path, map_location=device, weights_only=True)
        if isinstance(ckpt, dict) and "model" in ckpt:
            ckpt = ckpt["model"]
        if not isinstance(ckpt, dict):
            raise ValueError(f"Unrecognised weights format in {path}.")
        return ckpt

    def predict_prob(self, img: np.ndarray) -> np.ndarray:
        """Return H x W probability map in [0, 1]; CUDA OOM -> tiled retry -> CPU retry."""
        if not isinstance(img, np.ndarray) or img.ndim != 2 or img.size == 0:
            raise ValueError(f"U-Net input must be a non-empty H x W grayscale array, got "
                             f"{getattr(img, 'shape', type(img).__name__)}.")
        if img.dtype != np.uint8:
            raise ValueError(f"U-Net input must be uint8 (Preprocessor output), got {img.dtype}.")
        try:
            return self._forward(img)
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            _logger.warning("CUDA out of memory on %s image; retrying with %d px tiles", img.shape, TILE)
        _empty_cache()
        try:
            return self._tiled(img)
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            _logger.warning("Still out of memory with tiles; moving U-Net to CPU")
        _empty_cache()
        self._to_device(torch.device("cpu"))
        return self._tiled(img)

    def _forward(self, img: np.ndarray) -> np.ndarray:
        """One forward pass on an H x W uint8 crop (reflect-padded to a multiple of 32)."""
        h, w = img.shape
        pad_h, pad_w = (-h) % DIVISOR, (-w) % DIVISOR
        padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode="reflect")
        x = torch.from_numpy(padded).to(self.device, torch.float32).div_(255.0)
        x = x.expand(1, 3, *padded.shape)
        x = (x - self._mean) / self._std
        with torch.no_grad():
            prob = torch.sigmoid(self.model(x))[0, 0, :h, :w]
        return prob.cpu().numpy().astype(np.float32)

    def _tiled(self, img: np.ndarray) -> np.ndarray:
        """Overlapping TILE x TILE windows; overlaps averaged."""
        h, w = img.shape
        acc = np.zeros((h, w), np.float32)
        hits = np.zeros((h, w), np.float32)
        for y in _starts(h):
            for x in _starts(w):
                crop = img[y:y + TILE, x:x + TILE]
                acc[y:y + crop.shape[0], x:x + crop.shape[1]] += self._forward(crop)
                hits[y:y + crop.shape[0], x:x + crop.shape[1]] += 1.0
        return acc / hits


def _starts(n: int) -> List[int]:
    """Window start offsets covering [0, n) with TILE windows overlapping by OVERLAP."""
    if n <= TILE:
        return [0]
    starts = list(range(0, n - TILE, TILE - OVERLAP))
    return starts + [n - TILE]


def _is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def _empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

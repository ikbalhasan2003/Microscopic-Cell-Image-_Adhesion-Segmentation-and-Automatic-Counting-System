"""U-Net (ResNet34 encoder) wrapper: per-pixel cell probability map."""
from pathlib import Path
from typing import Dict

import numpy as np
import segmentation_models_pytorch as smp
import torch

from utils.logger import get_logger

# ImageNet normalisation; training/dataset.py imports these so train and inference match.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
DIVISOR = 32  # ResNet34 encoder downsamples 5x

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
        self.device = device
        self.model = build_unet()
        self.model.load_state_dict(self._read_state_dict(path, device))
        self.model.to(device).eval()
        self._mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
        _logger.info("U-Net loaded from %s on %s", path, device)

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
        """Return H x W probability map in [0, 1]."""
        if not isinstance(img, np.ndarray) or img.ndim != 2 or img.size == 0:
            raise ValueError(f"U-Net input must be a non-empty H x W grayscale array, got "
                             f"{getattr(img, 'shape', type(img).__name__)}.")
        if img.dtype != np.uint8:
            raise ValueError(f"U-Net input must be uint8 (Preprocessor output), got {img.dtype}.")
        h, w = img.shape
        pad_h, pad_w = (-h) % DIVISOR, (-w) % DIVISOR
        padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode="reflect")

        x = torch.from_numpy(padded).to(self.device, torch.float32).div_(255.0)
        x = x.expand(1, 3, *padded.shape)
        x = (x - self._mean) / self._std
        with torch.no_grad():
            prob = torch.sigmoid(self.model(x))[0, 0, :h, :w]
        return prob.cpu().numpy().astype(np.float32)

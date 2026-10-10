"""U-Net (ResNet34 encoder) wrapper: per-pixel cell probability map."""
import numpy as np
import torch

# ImageNet normalisation; training/dataset.py imports these so train and inference match.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class UNetSegmenter:
    def __init__(self, weights: str, device: torch.device) -> None:
        raise NotImplementedError

    def predict_prob(self, img: np.ndarray) -> np.ndarray:
        """Return H x W probability map in [0, 1]."""
        raise NotImplementedError

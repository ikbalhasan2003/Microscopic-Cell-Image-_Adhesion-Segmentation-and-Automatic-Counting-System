"""DSB 2018 loader: merge instance masks, resize 256x256, augment."""
import time
from pathlib import Path
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from core.preprocess import Preprocessor
from core.unet_model import MEAN, STD
from utils.logger import get_logger

SPLIT_SEED = 42
SPLIT_FRACTIONS = (0.8, 0.1)  # train, val; test = rest
SPLITS = ("train", "val", "test")
CLIP, BLUR_K = 2.0, 3         # same as inference defaults (config.Params)

_logger = get_logger(__name__)


def split_ids(root: str) -> dict:
    """Sorted sample ids -> deterministic 80/10/10 split {'train': [...], 'val': [...], 'test': [...]}."""
    path = Path(root)
    if not path.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {path}. Download Kaggle 'data-science-bowl-2018' "
            "(or the BBBC038 mirror) and pass the stage1_train/ folder.")
    ids = sorted(p.name for p in path.iterdir() if (p / "images").is_dir())
    if not ids:
        raise FileNotFoundError(f"No sample folders with images/ found in {path}.")
    order = np.random.default_rng(SPLIT_SEED).permutation(len(ids))
    n_train = int(SPLIT_FRACTIONS[0] * len(ids))
    n_val = int(SPLIT_FRACTIONS[1] * len(ids))
    shuffled = [ids[i] for i in order]
    return {"train": shuffled[:n_train], "val": shuffled[n_train:n_train + n_val],
            "test": shuffled[n_train + n_val:]}


def read_rgb(path: Path) -> np.ndarray:
    """Read PNG as RGB uint8 (alpha dropped, gray expanded)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image {path}.")
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)


def read_instances(sample_dir: Path) -> np.ndarray:
    """Instance label image (int32, 1..N) from masks/*.png."""
    labels = None
    for i, mp in enumerate(sorted((sample_dir / "masks").glob("*.png")), start=1):
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if labels is None:
            labels = np.zeros(m.shape, np.int32)
        labels[m > 0] = i
    if labels is None:
        raise ValueError(f"No masks found in {sample_dir / 'masks'}.")
    return labels


class DSB2018Dataset(Dataset):
    def __init__(self, root: str, split: str = "train", size: int = 256) -> None:
        """Load + preprocess + resize every sample of the split once (cached in RAM)."""
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}.")
        if int(size) != size or size < 32 or size % 32:
            raise ValueError(f"size must be a multiple of 32 and >= 32, got {size}.")
        self.root, self.split, self.size = Path(root), split, int(size)
        self.ids: List[str] = split_ids(root)[split]
        self._aug = self._build_aug() if split == "train" else None
        self._mean = np.array(MEAN, np.float32).reshape(3, 1, 1)
        self._std = np.array(STD, np.float32).reshape(3, 1, 1)
        t0 = time.perf_counter()
        pre = Preprocessor()
        self._images, self._masks = [], []
        for sid in self.ids:
            gray = pre.run(read_rgb(self.root / sid / "images" / f"{sid}.png"), clip=CLIP, blur_k=BLUR_K)
            mask = (read_instances(self.root / sid) > 0).astype(np.uint8)
            self._images.append(cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_LINEAR))
            self._masks.append(cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST))
        _logger.info("DSB2018 %s: %d samples cached in %.1f s", split, len(self.ids), time.perf_counter() - t0)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """(image 3 x S x S float normalised, mask 1 x S x S float {0, 1})."""
        img, mask = self._images[idx], self._masks[idx]
        if self._aug is not None:
            out = self._aug(image=img, mask=mask)
            img, mask = out["image"], out["mask"]
        x = np.repeat(img[None].astype(np.float32) / 255.0, 3, axis=0)
        x = (x - self._mean) / self._std
        return torch.from_numpy(x), torch.from_numpy(mask[None].astype(np.float32))

    @staticmethod
    def _build_aug() -> A.Compose:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ElasticTransform(alpha=50, sigma=7, border_mode=cv2.BORDER_REFLECT_101, p=0.2),
            A.RandomBrightnessContrast(p=0.3),
        ])

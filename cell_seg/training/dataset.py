"""DSB 2018 loader: merge instance masks, resize 256x256, augment."""
from typing import Tuple

import torch
from torch.utils.data import Dataset


class DSB2018Dataset(Dataset):
    def __init__(self, root: str, split: str = "train", size: int = 256) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

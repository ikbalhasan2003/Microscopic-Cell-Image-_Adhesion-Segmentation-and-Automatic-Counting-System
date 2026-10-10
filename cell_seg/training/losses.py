"""L = BCE + (1 - Dice)."""
import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

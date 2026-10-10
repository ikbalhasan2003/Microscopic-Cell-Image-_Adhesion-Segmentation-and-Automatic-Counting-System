"""L = BCE + (1 - Dice)."""
import torch
import torch.nn as nn

SMOOTH = 1.0


class BCEDiceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """BCEWithLogits + (1 - soft Dice on sigmoid(logits)); Dice averaged over the batch."""
        if logits.shape != target.shape:
            raise ValueError(f"logits {tuple(logits.shape)} and target {tuple(target.shape)} differ.")
        target = target.float()
        prob = torch.sigmoid(logits).flatten(1)
        flat = target.flatten(1)
        inter = (prob * flat).sum(1)
        dice = (2.0 * inter + SMOOTH) / (prob.sum(1) + flat.sum(1) + SMOOTH)
        return self.bce(logits, target) + (1.0 - dice.mean())

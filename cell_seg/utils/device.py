"""GPU -> CPU auto-fallback."""
import torch


def get_device() -> torch.device:
    raise NotImplementedError


def device_label(device: torch.device) -> str:
    """'CUDA (GPU)' or 'CPU' for the UI device card."""
    raise NotImplementedError

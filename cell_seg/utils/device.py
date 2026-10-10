"""GPU -> CPU auto-fallback."""
import torch

from utils.logger import get_logger

_logger = get_logger(__name__)
_logged = False


def get_device() -> torch.device:
    """CUDA if available, else CPU; logs the choice once per process."""
    global _logged
    # device_count() check: with CUDA_VISIBLE_DEVICES="" is_available() can be True with 0 devices.
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = torch.device("cuda" if use_cuda else "cpu")
    if not _logged:
        if device.type == "cuda":
            _logger.info("Using device: CUDA (%s)", torch.cuda.get_device_name(0))
        else:
            _logger.info("Using device: CPU (CUDA not available)")
        _logged = True
    return device


def device_label(device: torch.device) -> str:
    """'CUDA (GPU)' or 'CPU' for the UI device card."""
    if not isinstance(device, torch.device):
        raise ValueError(f"device must be torch.device, got {type(device).__name__}")
    return "CUDA (GPU)" if device.type == "cuda" else "CPU"

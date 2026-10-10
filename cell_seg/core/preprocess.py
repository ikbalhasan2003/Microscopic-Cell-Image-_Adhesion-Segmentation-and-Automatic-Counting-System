"""CLAHE contrast boost + Gaussian denoising."""
import cv2
import numpy as np

MIN_SIDE = 32


class Preprocessor:
    def run(self, img: np.ndarray, clip: float = 2.0, blur_k: int = 3) -> np.ndarray:
        """RGB/gray/RGBA (uint8, uint16 or float) -> H x W uint8 grayscale after CLAHE + blur."""
        self._validate(img, clip, blur_k)
        gray = self._to_gray(self._to_uint8(img))
        out = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(8, 8)).apply(gray)
        if blur_k > 1:
            k = blur_k if blur_k % 2 == 1 else blur_k + 1
            out = cv2.GaussianBlur(out, (k, k), 0)
        return out

    @staticmethod
    def _validate(img: np.ndarray, clip: float, blur_k: int) -> None:
        if not isinstance(img, np.ndarray) or img.size == 0:
            raise ValueError("Image is empty or not a numpy array.")
        if img.ndim not in (2, 3):
            raise ValueError(f"Image must be 2-D or 3-D, got shape {img.shape}.")
        if img.ndim == 3 and img.shape[2] not in (1, 3, 4):
            raise ValueError(f"Image must have 1, 3 or 4 channels, got {img.shape[2]}.")
        if min(img.shape[:2]) < MIN_SIDE:
            raise ValueError(f"Image too small: {img.shape[:2]}, need at least {MIN_SIDE}x{MIN_SIDE} px.")
        if not (img.dtype in (np.uint8, np.uint16) or np.issubdtype(img.dtype, np.floating)):
            raise ValueError(f"Unsupported dtype {img.dtype}; use uint8, uint16 or float.")
        if clip <= 0:
            raise ValueError(f"CLAHE clip limit must be > 0, got {clip}.")
        if int(blur_k) != blur_k:
            raise ValueError(f"blur_k must be an integer, got {blur_k}.")

    @staticmethod
    def _to_uint8(img: np.ndarray) -> np.ndarray:
        if img.dtype == np.uint8:
            return img
        if img.dtype == np.uint16:
            peak = int(img.max())
            if peak == 0:
                return np.zeros(img.shape, np.uint8)
            return (img.astype(np.float32) * (255.0 / peak)).round().astype(np.uint8)
        f = np.nan_to_num(img.astype(np.float32))
        if f.max() <= 1.0:
            f = np.clip(f, 0.0, 1.0) * 255.0
        return np.clip(f, 0.0, 255.0).round().astype(np.uint8)

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return img
        if img.shape[2] == 1:
            return img[:, :, 0]
        return cv2.cvtColor(np.ascontiguousarray(img[:, :, :3]), cv2.COLOR_RGB2GRAY)

"""CLAHE contrast boost + Gaussian denoising."""
import numpy as np


class Preprocessor:
    def run(self, img: np.ndarray, clip: float = 2.0, blur_k: int = 3) -> np.ndarray:
        raise NotImplementedError

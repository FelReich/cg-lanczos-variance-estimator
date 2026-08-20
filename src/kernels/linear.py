from __future__ import annotations

import numpy as np

from .base import Kernel


class LinearKernel(Kernel):
    def __init__(self, outputscale: float = 1.0, offset: float = 0.0,):
        if outputscale <= 0:
            raise ValueError("outputscale must be positive.")

        self.outputscale = float(outputscale)
        self.offset = float(offset)

    def matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = self._as_2d(X1)
        X2 = self._as_2d(X2)

        X1_centered = X1 - self.offset
        X2_centered = X2 - self.offset

        return self.outputscale**2 * (X1_centered @ X2_centered.T)
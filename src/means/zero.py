from __future__ import annotations

import numpy as np

from .base import Mean


class ZeroMean(Mean):
    def forward(self, X: np.ndarray) -> np.ndarray:
        X = self._as_2d(X)
        return np.zeros(X.shape[0])
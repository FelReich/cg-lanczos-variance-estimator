from __future__ import annotations

import numpy as np

from .base import Mean


class LinearMean(Mean):
    def __init__(self, weights: float | np.ndarray = 1.0, bias: float = 0.0,):
        self.weights = np.asarray(weights, dtype=float)
        self.bias = float(bias)

    def forward(self, X: np.ndarray) -> np.ndarray:
        X = self._as_2d(X)

        if self.weights.ndim == 0:
            return self.weights * X.squeeze(-1) + self.bias

        if self.weights.shape != (X.shape[1],):
            raise ValueError("weights must be a scalar or have shape equal to the input dimension.")

        return X @ self.weights + self.bias
from __future__ import annotations

import torch

from .base import Mean


class LinearMean(Mean):
    def __init__(self, weights: float | torch.Tensor = 1.0, bias: float = 0.0):
        self.weights = torch.as_tensor(weights)
        self.bias = float(bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self._as_2d(X)

        weights = self.weights.to(dtype=X.dtype, device=X.device)

        if weights.ndim == 0:
            return weights * X.squeeze(-1) + self.bias

        if weights.shape != (X.shape[1],):
            raise ValueError("weights must be a scalar or have shape equal to the input dimension.")

        return X @ weights + self.bias
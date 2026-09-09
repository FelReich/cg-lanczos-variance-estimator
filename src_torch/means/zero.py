from __future__ import annotations

import torch

from .base import Mean


class ZeroMean(Mean):
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self._as_2d(X)
        return torch.zeros(X.shape[0], dtype=X.dtype, device=X.device)
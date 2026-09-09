from __future__ import annotations

import torch

from .base import Kernel


class RBFKernel(Kernel):
    def __init__(self, lengthscale: float = 1.0, outputscale: float = 1.0):
        if lengthscale <= 0:
            raise ValueError("lengthscale must be positive.")
        if outputscale <= 0:
            raise ValueError("outputscale must be positive.")

        self.lengthscale = float(lengthscale)
        self.outputscale = float(outputscale)

    def matrix(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        sq_dist = self._dist(X1, X2, squared=True)

        return self.outputscale**2 * torch.exp(
            -sq_dist / (2.0 * self.lengthscale**2)
        )
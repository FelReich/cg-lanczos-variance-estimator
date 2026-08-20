from __future__ import annotations

import numpy as np

from .base import Kernel


class MaternKernel(Kernel):
    def __init__(self, lengthscale: float = 1.0, outputscale: float = 1.0, nu: float = 2.5,):
        if lengthscale <= 0:
            raise ValueError("lengthscale must be positive.")
        if outputscale <= 0:
            raise ValueError("outputscale must be positive.")
        if nu not in {0.5, 1.5, 2.5}:
            raise ValueError("nu must be one of 0.5, 1.5, or 2.5.")

        self.lengthscale = float(lengthscale)
        self.outputscale = float(outputscale)
        self.nu = float(nu)

    def matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        dist = self._dist(X1, X2)
        r = dist / self.lengthscale

        if self.nu == 0.5:
            constant = 1.0
            exp_component = np.exp(-r)

        elif self.nu == 1.5:
            sqrt3_r = np.sqrt(3.0) * r
            constant = 1.0 + sqrt3_r
            exp_component = np.exp(-sqrt3_r)

        else:
            sqrt5_r = np.sqrt(5.0) * r
            constant = 1.0 + sqrt5_r + (5.0 / 3.0) * r**2
            exp_component = np.exp(-sqrt5_r)

        return self.outputscale**2 * constant * exp_component
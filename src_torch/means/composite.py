from __future__ import annotations

import torch

from .base import Mean


class TrigExpMean(Mean):
    # One-dimensional mean function combining sine, cosine, and Gaussian-shaped exponential terms.

    def __init__(
        self,
        offset: float = 0.0,
        sine_scale: float = 1.0,
        sine_frequency: float = 1.0,
        cosine_scale: float = 1.0,
        cosine_frequency: float = 1.0,
        exp_scale: float = 1.0,
        exp_rate: float = 1.0,
    ):
        self.offset = float(offset)
        self.sine_scale = float(sine_scale)
        self.sine_frequency = float(sine_frequency)
        self.cosine_scale = float(cosine_scale)
        self.cosine_frequency = float(cosine_frequency)
        self.exp_scale = float(exp_scale)
        self.exp_rate = exp_rate

    @property
    def exp_rate(self):
        return self._exp_rate

    @exp_rate.setter
    def exp_rate(self, exp_rate):
        exp_rate = float(exp_rate)

        if exp_rate <= 0:
            raise ValueError("exp_rate must be positive.")

        self._exp_rate = exp_rate

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self._as_2d(X)

        if X.shape[1] != 1:
            raise ValueError("TrigExpMean currently supports one-dimensional inputs only.")

        x = X.squeeze(-1)

        return (
            self.offset
            + self.sine_scale * torch.sin(self.sine_frequency * x)
            + self.cosine_scale * torch.cos(self.cosine_frequency * x)
            + self.exp_scale * torch.exp(-self.exp_rate * x**2)
        )
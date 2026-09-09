from __future__ import annotations

import torch


class Mean:
    """Base class for mean functions.

    Mean functions map input locations to prior mean values and are evaluated by
    calling the object directly.

    :param torch.Tensor X: Input locations of shape `n x d`.
    :return: Mean values of shape `n`.
    """

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return self.forward(X)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _as_2d(X: torch.Tensor) -> torch.Tensor:
        X = torch.as_tensor(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("Input arrays must be one- or two-dimensional.")

        return X
from __future__ import annotations

import torch


class Kernel:
    """Base class for covariance kernels.

    Kernel objects map two sets of input locations to their covariance matrix and
    are evaluated by calling the object directly.

    :param torch.Tensor X1: First input matrix of shape `n x d`.
    :param torch.Tensor X2: Second input matrix of shape `m x d`.
    :return: Covariance matrix of shape `n x m`.
    """

    def __call__(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        return self.matrix(X1, X2)

    def matrix(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _as_2d(X: torch.Tensor) -> torch.Tensor:
        X = torch.as_tensor(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("Input arrays must be one- or two-dimensional.")

        return X

    def _dist(self, X1: torch.Tensor, X2: torch.Tensor, squared: bool = False) -> torch.Tensor:
        # Pairwise Euclidean distances.
        X1 = self._as_2d(X1)
        X2 = self._as_2d(X2)

        diff = X1[:, None, :] - X2[None, :, :]
        sq_dist = torch.sum(diff**2, dim=-1)

        if squared:
            return sq_dist

        return torch.sqrt(sq_dist)
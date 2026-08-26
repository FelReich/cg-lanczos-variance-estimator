from __future__ import annotations

import numpy as np


class Mean:
    """Base class for mean functions.

    Mean functions map input locations to prior mean values and are evaluated by
    calling the object directly.

    :param numpy.ndarray X: Input locations of shape `n x d`.
    :return: Mean values of shape `n`.
    """
    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def forward(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _as_2d(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("Input arrays must be one- or two-dimensional.")

        return X
from __future__ import annotations

import numpy as np


class Kernel:
    """Base class for covariance kernels.

    Kernel objects map two sets of input locations to their covariance matrix and
    are evaluated by calling the object directly.

    :param numpy.ndarray X1: First input matrix of shape `n x d`.
    :param numpy.ndarray X2: Second input matrix of shape `m x d`.
    :return: Covariance matrix of shape `n x m`.
    """
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return self.matrix(X1, X2)

    def matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _as_2d(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("Input arrays must be one- or two-dimensional.")

        return X
    
    def _dist(self, X1: np.ndarray, X2: np.ndarray, squared: bool = False):
        #Pairwise squared Euclidean distances
        X1 = self._as_2d(X1)
        X2 = self._as_2d(X2)

        diff = X1[:, None, :] - X2[None, :, :]
        sq_dist = np.sum(diff**2, axis=-1)

        if squared:
            return sq_dist
        
        return np.sqrt(sq_dist)
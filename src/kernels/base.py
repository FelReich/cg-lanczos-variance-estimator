from __future__ import annotations

import numpy as np


class Kernel:
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
        X1 = self._as_2d(X1)
        X2 = self._as_2d(X2)

        diff = X1[:, None, :] - X2[None, :, :]
        sq_dist = np.sum(diff**2, axis=-1)

        if squared:
            return sq_dist
        
        return np.sqrt(sq_dist)
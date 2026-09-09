from __future__ import annotations

import torch

from .corrections import exact_correction, love_correction
from .linalg.cg import linear_cg, cg_store_lanczos_basis
from .linalg.lanczos import lanczos_tridiag, extend_lanczos_basis


class GP:
    """A minimal PyTorch Gaussian process model for LOVE and CG-Lanczos extension tests."""

    def __init__(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        kernel,
        mean,
        noise: float = 1e-2,
    ):
        if noise < 0:
            raise ValueError("noise must be nonnegative.")

        self.X_train = self._as_2d(X_train)
        self.y_train = torch.as_tensor(
            y_train,
            dtype=self.X_train.dtype,
            device=self.X_train.device,
        ).reshape(-1)

        self.kernel = kernel
        self.mean = mean
        self.noise = float(noise)

        if self.X_train.shape[0] != self.y_train.shape[0]:
            raise ValueError("X_train and y_train must have matching first dimension.")

        self.K = self.kernel(self.X_train, self.X_train)

        eye = torch.eye(
            self.X_train.shape[0],
            dtype=self.X_train.dtype,
            device=self.X_train.device,
        )
        self.K_noise = self.K + self.noise * eye

        self.train_mean = self.mean(self.X_train)
        self.centered_y = self.y_train - self.train_mean

        self.method = None
        self.alpha = None
        self.Q = None
        self.T = None
        self.jitter = None

    @staticmethod
    def _as_2d(X: torch.Tensor) -> torch.Tensor:
        X = torch.as_tensor(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("Input arrays must be one- or two-dimensional.")

        return X

    def _matmul_closure(self, v: torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.K_noise, v)

    def _rhs(self) -> torch.Tensor:
        return self.centered_y.reshape(1, -1, 1)

    def compute_posterior(
        self,
        method: str = "exact",
        cg_J: int | None = None,
        lanczos_J: int | None = None,
        tol: float = 1e-6,
        jitter: float = 1e-6,
    ) -> None:
        self.method = method.lower()
        self.jitter = float(jitter)

        self.alpha = None
        self.Q = None
        self.T = None

        rhs = self._rhs()

        match self.method:
            case "exact":
                self.alpha = torch.linalg.solve(self.K_noise, self.centered_y)

            case "love":
                if cg_J is None:
                    raise ValueError("cg_J must be specified for method='love'.")
                if lanczos_J is None:
                    raise ValueError("lanczos_J must be specified for method='love'.")

                alpha = linear_cg(
                    self._matmul_closure,
                    rhs,
                    tolerance=tol,
                    max_iter=cg_J,
                )
                self.alpha = alpha.reshape(-1)

                self.Q, self.T = lanczos_tridiag(
                    self._matmul_closure,
                    max_iter=lanczos_J,
                    dtype=self.K_noise.dtype,
                    device=self.K_noise.device,
                    matrix_shape=self.K_noise.shape,
                    batch_shape=rhs.shape[:-2],
                    init_vecs=rhs,
                    num_init_vecs=1,
                    tol=tol,
                )

            case "extend":
                if cg_J is None:
                    raise ValueError("cg_J must be specified for method='extend'.")
                if lanczos_J is None:
                    raise ValueError("lanczos_J must be specified for method='extend'.")

                alpha, Q, T = cg_store_lanczos_basis(
                    self._matmul_closure,
                    rhs,
                    tolerance=tol,
                    max_iter=cg_J,
                )
                self.alpha = alpha.reshape(-1)

                self.Q, self.T = extend_lanczos_basis(
                    self._matmul_closure,
                    max_iter=lanczos_J,
                    dtype=self.K_noise.dtype,
                    device=self.K_noise.device,
                    matrix_shape=self.K_noise.shape,
                    q_mat=Q,
                    t_mat=T,
                    tol=tol,
                )

            case _:
                raise ValueError("method must be one of 'exact', 'love', or 'extend'.")

    def _require_posterior(self) -> None:
        if self.method is None or self.alpha is None:
            raise RuntimeError("compute_posterior must be called before prediction.")

    def prior_covariance(self, X_test: torch.Tensor) -> torch.Tensor:
        X_test = self._as_2d(X_test).to(
            dtype=self.X_train.dtype,
            device=self.X_train.device,
        )
        return self.kernel(X_test, X_test)

    def train_test_covariance(self, X_test: torch.Tensor) -> torch.Tensor:
        X_test = self._as_2d(X_test).to(
            dtype=self.X_train.dtype,
            device=self.X_train.device,
        )
        return self.kernel(self.X_train, X_test)

    def predict_mean(self, X_test: torch.Tensor) -> torch.Tensor:
        self._require_posterior()

        X_test = self._as_2d(X_test).to(
            dtype=self.X_train.dtype,
            device=self.X_train.device,
        )
        k = self.train_test_covariance(X_test)

        return self.mean(X_test) + k.transpose(-1, -2) @ self.alpha

    def predict_covariance(self, X_test: torch.Tensor) -> torch.Tensor:
        self._require_posterior()

        X_test = self._as_2d(X_test).to(
            dtype=self.X_train.dtype,
            device=self.X_train.device,
        )

        K_test = self.prior_covariance(X_test)
        k = self.train_test_covariance(X_test)

        match self.method:
            case "exact":
                correction = exact_correction(self.K_noise, k)

            case "love" | "extend":
                if self.Q is None or self.T is None:
                    raise RuntimeError("Lanczos posterior has not been computed.")

                Q = self.Q.squeeze(0)
                T = self.T.squeeze(0)

                correction = love_correction(
                    Q,
                    T,
                    k,
                    jitter=self.jitter,
                )

            case _:
                raise RuntimeError("Invalid posterior method state.")

        covariance = K_test - correction
        return 0.5 * (covariance + covariance.transpose(-1, -2))

    def predict_variance(self, X_test: torch.Tensor) -> torch.Tensor:
        covariance = self.predict_covariance(X_test)
        return torch.diag(covariance)

    def predict(self, X_test: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.predict_mean(X_test)
        variance = self.predict_variance(X_test)

        return mean, variance
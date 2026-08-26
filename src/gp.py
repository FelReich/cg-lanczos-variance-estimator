from __future__ import annotations

import numpy as np

from src.linalg import ReorthogonalizationRule, cg, lanczos_tridiagonalization
from src.corrections import (
    exact_correction,
    cg_cholesky_correction,
    cg_qr_correction,
    love_correction,
)


class GP:
    """A minimal Gaussian process regression model for posterior covariance experiments.

    This class stores training inputs, training targets, a kernel, and a mean function.
    It constructs the noisy training covariance matrix
    :math:`\widehat K = K_{XX} + \sigma^2 I` and provides posterior predictions using
    exact inference, CG-based covariance approximations, or LOVE.

    :param numpy.ndarray X_train: Training inputs of shape `n x d`.
    :param numpy.ndarray y_train: Training targets of shape `n`.
    :param Kernel kernel: Kernel used to construct prior covariance matrices.
    :param Mean mean: Mean function used to center the observations and evaluate predictive means.
    :param float noise: Observation noise added to the training covariance matrix. (Default: `1e-2`.)

    :ivar numpy.ndarray X_train: Stored training inputs.
    :ivar numpy.ndarray y_train: Stored training targets.
    :ivar Kernel kernel: Kernel used by the GP.
    :ivar Mean mean: Mean function used by the GP.
    :ivar float noise: Observation noise level.
    :ivar numpy.ndarray K: Prior training covariance matrix `K_{XX}`.
    :ivar numpy.ndarray K_noise: Noisy training covariance matrix `K_{XX} + noise * I`.
    :ivar numpy.ndarray train_mean: Mean function evaluated at the training inputs.
    :ivar numpy.ndarray centered_y: Centered observations `y_train - train_mean`.
    :ivar str method: Posterior approximation method selected by :meth:`compute_posterior`.
    :ivar numpy.ndarray alpha: Posterior mean weights solving approximately `K_noise @ alpha = centered_y`.
    :ivar numpy.ndarray D: Stored CG search directions for the CG covariance approximation.
    :ivar numpy.ndarray KD: Matrix-vector products `K_noise @ D`.
    :ivar numpy.ndarray Q: Stored Lanczos basis for LOVE.
    :ivar numpy.ndarray T: Projected Lanczos matrix `Q.T @ K_noise @ Q`.
    :ivar float jitter: Jitter used in approximate covariance corrections.
    :ivar bool coordinate_jitter: Whether CG-Cholesky uses coordinate jitter or operator-consistent jitter.
    """

    def __init__(self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        kernel,
        mean,
        noise: float = 1e-2,
    ):
        if noise < 0:
            raise ValueError("noise must be nonnegative.")

        self.X_train = self._as_2d(X_train)
        self.y_train = np.asarray(y_train, dtype=float).reshape(-1)
        self.kernel = kernel
        self.mean = mean
        self.noise = float(noise)

        if self.X_train.shape[0] != self.y_train.shape[0]:
            raise ValueError("X_train and y_train must have matching first dimension.")

        self.K = self.kernel(self.X_train, self.X_train)
        self.K_noise = self.K + self.noise * np.eye(self.X_train.shape[0])

        self.train_mean = self.mean(self.X_train)
        self.centered_y = self.y_train - self.train_mean

        self.method = None
        self.cg_correction_method = None

        self.cg_reorthogonalization_rule = None
        self.lanczos_reorthogonalize = None

        self.alpha = None

        self.D = None
        self.KD = None

        self.Q = None
        self.T = None

        self.jitter = None
        self.coordinate_jitter = None

        

    @staticmethod
    def _as_2d(X: np.ndarray) -> np.ndarray:
        #Converts inputs to a two-dimensional NumPy array
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("Input arrays must be one- or two-dimensional.")

        return X
    
    def compute_posterior(
        self,
        method: str = "exact",
        cg_J: int | None = None,
        lanczos_J: int | None = None,
        tol: float = 1e-6,
        jitter: float = 1e-6,
        cg_reorthogonalization_rule: ReorthogonalizationRule | None = None,
        lanczos_reorthogonalize: bool = True,
        coordinate_jitter: bool = True,
    ) -> None:
        """Computes posterior quantities needed for prediction.

        For `method="exact"`, the linear system with the noisy training covariance
        is solved directly. For `method="cg"`, the same system is solved with CG and
        the CG search directions are stored for the covariance correction. For
        `method="love"`, CG is used for the posterior mean solve and a separate
        Lanczos basis is constructed for the covariance correction.

        :param str method: Posterior approximation method. Must be `"exact"`, `"cg"`, or `"love"`. (Default: `"exact"`.)
        :param int cg_J: Number of CG iterations used for `"cg"` and `"love"`.
        :param int lanczos_J: Number of Lanczos iterations used for `"love"`.
        :param float tol: Relative residual tolerance for iterative methods. (Default: `1e-6`.)
        :param float jitter: Jitter used in approximate covariance corrections. (Default: `1e-6`.)
        :param ReorthogonalizationRule cg_reorthogonalization_rule: Optional reorthogonalization rule for CG search directions.
        :param bool lanczos_reorthogonalize: If True, reorthogonalize Lanczos vectors. (Default: True.)
        :param bool coordinate_jitter: If True, use coordinate jitter in the CG-Cholesky correction. (Default: True.)
        """

        self.method = method.lower()

        self.jitter = float(jitter)
        self.coordinate_jitter = bool(coordinate_jitter)

        self.cg_reorthogonalization_rule = cg_reorthogonalization_rule
        self.lanczos_reorthogonalize = bool(lanczos_reorthogonalize)

        self.D = None
        self.KD = None
        self.Q = None
        self.T = None

        match self.method:
            case "exact":
                self.alpha = np.linalg.solve(self.K_noise, self.centered_y)

            case "cg":
                if cg_J is None:
                    raise ValueError("cg_J must be specified for method='cg'.")

                self.alpha, self.D, self.KD = cg(
                    lambda v: self.K_noise @ v,
                    self.centered_y,
                    J=cg_J,
                    tol=tol,
                    save_directions=True,
                    reorthogonalization_rule=cg_reorthogonalization_rule,
                )

            case "love":
                if cg_J is None:
                    raise ValueError("cg_J must be specified for method='love'.")
                if lanczos_J is None:
                    raise ValueError("lanczos_J must be specified for method='love'.")

                self.alpha = cg(
                    lambda v: self.K_noise @ v,
                    self.centered_y,
                    J=cg_J,
                    tol=tol,
                    save_directions=False,
                    reorthogonalization_rule=None,
                )

                self.Q, self.T = lanczos_tridiagonalization(
                    lambda v: self.K_noise @ v,
                    self.centered_y,
                    num_iter=lanczos_J,
                    tol=tol,
                    reorthogonalize=lanczos_reorthogonalize,
                )

            case _:
                raise ValueError("method must be one of 'exact', 'cg', or 'love'.")
            
    def _require_posterior(self) -> None:
        #Checks that posterior quantities have already been computed
        if self.method is None or self.alpha is None:
            raise RuntimeError("compute_posterior must be called before prediction.")

    def prior_covariance(self, X_test: np.ndarray) -> np.ndarray:
        #Computes the prior covariance matrix `K_{X_*X_*}`
        X_test = self._as_2d(X_test)
        return self.kernel(X_test, X_test)
    
    def train_test_covariance(self, X_test: np.ndarray) -> np.ndarray:
        #Computes the train-test covariance matrix `K_{XX_*}`
        X_test = self._as_2d(X_test)
        return self.kernel(self.X_train, X_test)
    
    def predict_mean(self, X_test: np.ndarray) -> np.ndarray:
        #Computes the posterior mean at the test inputs
        self._require_posterior()

        X_test = self._as_2d(X_test)
        k = self.train_test_covariance(X_test)

        return self.mean(X_test) + k.T @ self.alpha
            
    def predict_covariance(self, X_test: np.ndarray, cg_correction_method: str = "qr") -> np.ndarray:
        """Computes the posterior covariance matrix at the test inputs.

        For `method="exact"`, the covariance correction is computed by a direct solve.
        For `method="cg"`, the correction is computed from stored CG search directions.
        In this case, `cg_correction_method="cholesky"` uses the raw CG direction basis,
        while `cg_correction_method="qr"` first transforms the stored directions to an
        orthonormal basis. For `method="love"`, the correction is computed from the
        stored Lanczos basis and tridiagonal matrix.

        :param numpy.ndarray X_test: Test inputs of shape `m x d`.
        :param str cg_correction_method: Correction method used only for `method="cg"`.
            Must be either `"cholesky"` or `"qr"`. (Default: `"cholesky"`.)
        :return: Posterior covariance matrix of shape `m x m`.
        """
        cg_correction_method = cg_correction_method.lower()
        if cg_correction_method not in {"cholesky", "qr"}:
            raise ValueError("cg_correction_method must be either 'cholesky' or 'qr'.")
        
        self.cg_correction_method = cg_correction_method
        self._require_posterior()

        K_test = self.prior_covariance(X_test)
        k = self.train_test_covariance(X_test)

        match self.method:
            case "exact":
                correction = exact_correction(self.K_noise, k)

            case "cg":
                if self.D is None or self.KD is None:
                    raise RuntimeError("CG posterior has not been computed.")
                
                match self.cg_correction_method:
                    case "cholesky":
                        correction = cg_cholesky_correction(
                            self.D,
                            self.KD,
                            k,
                            jitter=self.jitter,
                            coordinate_jitter=self.coordinate_jitter,
                        )
                    case "qr":
                        correction = cg_qr_correction(
                            self.D,
                            self.KD,
                            k,
                            jitter=self.jitter,
                        )

            case "love":
                if self.Q is None or self.T is None:
                    raise RuntimeError("LOVE posterior has not been computed.")
                
                correction = love_correction(
                    self.Q,
                    self.T,
                    k,
                    jitter=self.jitter,
                )
            
            case _:
                raise RuntimeError("Invalid posterior method state.")
            
        covariance = K_test - correction
        return 0.5 * (covariance + covariance.T)
    
    def predict_variance(self, X_test: np.ndarray, cg_correction_method: str = "cholesky") -> np.ndarray:
        #Computes the marginal posterior variances at the test inputs
        covariance = self.predict_covariance(X_test, cg_correction_method=cg_correction_method)

        return np.diag(covariance)

    def predict(self, X_test: np.ndarray, cg_correction_method: str = "cholesky") -> tuple[np.ndarray, np.ndarray]:
        #Computes posterior mean and marginal posterior variances at the test inputs
        mean = self.predict_mean(X_test)
        variance = self.predict_variance(X_test, cg_correction_method=cg_correction_method)
        
        return mean, variance
        
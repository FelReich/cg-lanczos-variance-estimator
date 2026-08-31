import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.corrections import cg_cholesky_correction, exact_correction, love_correction
from src.diagnostics import conjugacy_error, projection_residual, relative_error
from src.gp import GP
from src.kernels import RBFKernel
from src.linalg import (
    ReorthogonalizationRule,
    cg,
    cg_store_lanczos_basis,
    extend_lanczos_basis,
    lanczos_tridiagonalization,
)
from src.means import ZeroMean


def compare_cholesky_negative_control(
    *,
    n: int = 1000,
    m: int = 100,
    cg_J: int = 100,
    lanczos_J: int = 100,
    outputscale: float = 1.0,
    lengthscales: list[float] | None = None,
    noises: list[float] | None = None,
    jitters: list[float] | None = None,
    domain: tuple[float, float] = (-10.0, 10.0),
    seed: int = 123,
    extension_tol: float = 1e-14,
    coordinate_jitter: bool = True,
) -> None:
    """Compares raw CG-Cholesky correction against Extend and LOVE.

    This script is intended as a negative control. It documents cases where the
    raw search-direction Gram matrix `D.T @ KD` becomes numerically unreliable,
    making the Cholesky-based correction less stable than the QR-based extended
    Lanczos construction.

    :param int n: Number of training points. (Default: `1000`.)
    :param int m: Number of test points. (Default: `100`.)
    :param int cg_J: Maximum number of CG iterations. (Default: `100`.)
    :param int lanczos_J: Maximum number of Lanczos iterations for LOVE. (Default: `100`.)
    :param float outputscale: Kernel outputscale. (Default: `1.0`.)
    :param list lengthscales: Lengthscales used in the experiment.
    :param list noises: Noise levels used in the experiment.
    :param list jitters: Jitter values used in approximate covariance corrections.
    :param tuple domain: Interval from which inputs are sampled. (Default: `(-10.0, 10.0)`.)
    :param int seed: Random seed. (Default: `123`.)
    :param float extension_tol: Breakdown tolerance for basis extension. (Default: `1e-14`.)
    :param bool coordinate_jitter: If True, regularize `D.T @ KD` by adding `jitter * I`.
    """
    if lengthscales is None:
        lengthscales = [0.1, 0.3, 1.0, 3.0, 10.0]

    if noises is None:
        noises = [1e-6, 1e-4, 1e-2, 1.0]

    if jitters is None:
        jitters = [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]

    rng = np.random.default_rng(seed)

    X_train = rng.uniform(domain[0], domain[1], size=(n, 1))
    X_test = rng.uniform(domain[0], domain[1], size=(m, 1))

    def f(x):
        return np.sin(x) + np.cos(3 * x) + np.exp(-x**2)

    y_train = f(X_train).reshape(-1)
    mean = ZeroMean()

    header = (
        f"{'lengthscale':<10} "
        f"{'noise':<10} "
        f"{'jitter':<9} "
        f"{'J_chol':<7} "
        f"{'J_ext':<6} "
        f"{'J_love':<7} "
        f"{'rel_chol_exact':<16} "
        f"{'rel_ext_exact':<15} "
        f"{'rel_love_exact':<16} "
        f"{'diag_chol_exact':<17} "
        f"{'diag_ext_exact':<16} "
        f"{'diag_love_exact':<17} "
        f"{'conj_D':<12} "
        f"{'proj_chol':<12} "
        f"{'proj_ext':<12} "
        f"{'proj_love':<12}"
    )
    print(header)
    print("-" * len(header))

    for lengthscale in lengthscales:
        kernel = RBFKernel(lengthscale=lengthscale, outputscale=outputscale)

        for noise in noises:
            try:
                gp = GP(X_train, y_train, kernel=kernel, mean=mean, noise=noise)
                gp.compute_posterior(method="exact")

                K_test = gp.prior_covariance(X_test)
                k = gp.train_test_covariance(X_test)

                exact_corr = exact_correction(gp.K_noise, k)
                exact_cov = K_test - exact_corr
                exact_cov = 0.5 * (exact_cov + exact_cov.T)
                exact_var = np.diag(exact_cov)

                rule = ReorthogonalizationRule(mode="always")

                _, D_chol, KD_chol = cg(
                    lambda v: gp.K_noise @ v,
                    gp.centered_y,
                    J=cg_J,
                    tol=1e-6,
                    save_directions=True,
                    reorthogonalization_rule=rule,
                )

                _, Q_resid, KQ_resid = cg_store_lanczos_basis(
                    lambda v: gp.K_noise @ v,
                    gp.centered_y,
                    J=cg_J,
                    tol=1e-6,
                    reorthogonalization_rule=rule,
                )

                Q_love, T_love = lanczos_tridiagonalization(
                    lambda v: gp.K_noise @ v,
                    gp.centered_y,
                    num_iter=lanczos_J,
                    tol=1e-6,
                    reorthogonalize=True,
                )

                target_J = max(Q_resid.shape[1], Q_love.shape[1])

                Q_ext, T_ext = extend_lanczos_basis(
                    lambda v: gp.K_noise @ v,
                    Q_resid,
                    KQ_resid,
                    target_J=target_J,
                    tol=extension_tol,
                )

                conj_D = conjugacy_error(D_chol, KD_chol)
                proj_chol = projection_residual(D_chol, k)
                proj_ext = projection_residual(Q_ext, k)
                proj_love = projection_residual(Q_love, k)

                for jitter in jitters:
                    rel_chol = np.nan
                    diag_chol = np.nan


                    chol_corr = cg_cholesky_correction(D_chol, KD_chol, k, jitter=jitter, coordinate_jitter=coordinate_jitter)  

                    chol_cov = K_test - chol_corr
                    chol_cov = 0.5 * (chol_cov + chol_cov.T)
                    chol_var = np.diag(chol_cov)

                    rel_chol = relative_error(chol_cov, exact_cov)
                    diag_chol = relative_error(chol_var, exact_var)

                    ext_corr = love_correction(Q_ext, T_ext, k, jitter=jitter)
                    love_corr = love_correction(Q_love, T_love, k, jitter=jitter)

                    ext_cov = K_test - ext_corr
                    love_cov = K_test - love_corr

                    ext_cov = 0.5 * (ext_cov + ext_cov.T)
                    love_cov = 0.5 * (love_cov + love_cov.T)

                    ext_var = np.diag(ext_cov)
                    love_var = np.diag(love_cov)

                    print(
                        f"{lengthscale:<10.1g} "
                        f"{noise:<10.1e} "
                        f"{jitter:<9.1e} "
                        f"{D_chol.shape[1]:<7d} "
                        f"{Q_ext.shape[1]:<6d} "
                        f"{Q_love.shape[1]:<7d} "
                        f"{rel_chol:<16.3e} "
                        f"{relative_error(ext_cov, exact_cov):<15.3e} "
                        f"{relative_error(love_cov, exact_cov):<16.3e} "
                        f"{diag_chol:<17.3e} "
                        f"{relative_error(ext_var, exact_var):<16.3e} "
                        f"{relative_error(love_var, exact_var):<17.3e} "
                        f"{conj_D:<12.3e} "
                        f"{proj_chol:<12.3e} "
                        f"{proj_ext:<12.3e} "
                        f"{proj_love:<12.3e}"
                    )
   

            except Exception as e:
                print(
                    f"{lengthscale:<10.1g} "
                    f"{noise:<10.1e} "
                    f"FAILED during setup: {type(e).__name__}: {e}"
                )


if __name__ == "__main__":
    compare_cholesky_negative_control()
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.corrections import cg_qr_correction, love_correction
from src.diagnostics import projection_residual, relative_error
from src.gp import GP
from src.kernels import RBFKernel
from src.linalg import ReorthogonalizationRule, cg, lanczos_tridiagonalization
from src.linalg.cg import cg_store_residuals
from src.linalg.lanczos import extend_lanczos_basis
from src.means import ZeroMean


def make_data(seed: int, n: int, m: int, domain: tuple[float, float]):
    rng = np.random.default_rng(seed)

    X_train = rng.uniform(domain[0], domain[1], (n, 1))
    X_test = rng.uniform(domain[0], domain[1], (m, 1))

    def f(x):
        return np.sin(x) + np.cos(3.0 * x) + np.exp(-x**2)

    y_train = f(X_train).reshape(-1)

    return X_train, y_train, X_test


def run_residual_lanczos_comparison(
    *,
    seed: int = 123,
    n: int = 1000,
    m: int = 100,
    domain: tuple[float, float] = (-3.0, 3.0),
    cg_J: int = 100,
    lanczos_J: int = 100,
    outputscale: float = 1.0,
    lengthscales: list[float] | None = None,
    noises: list[float] | None = None,
    jitters: list[float] | None = None,
    cg_reorthogonalization_mode: str = "always",
    cg_reorthogonalization_every: int = 1,
    cg_reorthogonalization_start: int = 1,
    cg_rayleigh_tol: float = 1.0,
    cg_norm_tol: float = 1.0,
):
    if lengthscales is None:
        lengthscales = [0.1, 0.3, 1.0, 3.0, 10.0]

    if noises is None:
        noises = [1e-6, 1e-4, 1e-2, 1.0]

    if jitters is None:
        jitters = [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]

    rule = ReorthogonalizationRule(
        mode=cg_reorthogonalization_mode,
        every=cg_reorthogonalization_every,
        start=cg_reorthogonalization_start,
        rayleigh_tol=cg_rayleigh_tol,
        norm_tol=cg_norm_tol,
    )

    X_train, y_train, X_test = make_data(seed, n, m, domain)
    mean = ZeroMean()

    header = (
        "lengthscale noise      jitter    "
        "J_qr  J_resid J_ext  J_love "
        "rel_qr_exact rel_resid_exact rel_ext_exact rel_love_exact "
        "diag_qr_exact diag_resid_exact diag_ext_exact diag_love_exact "
        "proj_qr     proj_resid   proj_ext     proj_love"
    )
    print(header)
    print("-" * len(header))

    for lengthscale in lengthscales:
        kernel = RBFKernel(lengthscale=lengthscale, outputscale=outputscale)

        for noise in noises:
            try:
                gp_exact = GP(X_train, y_train, kernel=kernel, mean=mean, noise=noise)
                gp_exact.compute_posterior(method="exact")

                exact_cov = gp_exact.predict_covariance(X_test)
                exact_var = np.diag(exact_cov)

                K_test = gp_exact.prior_covariance(X_test)
                k = gp_exact.train_test_covariance(X_test)

                _, D_qr, KD_qr = cg(
                    lambda v: gp_exact.K_noise @ v,
                    gp_exact.centered_y,
                    J=cg_J,
                    tol=1e-6,
                    save_directions=True,
                    reorthogonalization_rule=rule,
                )

                _, Q_resid, T_resid = cg_store_residuals(
                    lambda v: gp_exact.K_noise @ v,
                    gp_exact.centered_y,
                    J=cg_J,
                    tol=1e-6,
                    save_residuals=True,
                )

                Q_love, T_love = lanczos_tridiagonalization(
                    lambda v: gp_exact.K_noise @ v,
                    gp_exact.centered_y,
                    num_iter=lanczos_J,
                    tol=1e-6,
                    reorthogonalize=True,
                )

                """if Q_resid.shape[1] < Q_love.shape[1]:
                    Q_ext, T_ext = extend_lanczos_basis(
                        lambda v: gp_exact.K_noise @ v,
                        Q_resid,
                        target_J=Q_love.shape[1],
                        tol=1e-12,
                        reorthogonalize=True,
                    )
                else:
                    Q_ext = Q_resid[:, : Q_love.shape[1]]
                    T_ext = Q_ext.T @ gp_exact.K_noise @ Q_ext
                    T_ext = 0.5 * (T_ext + T_ext.T)"""
                
                Q_ext, T_ext = extend_lanczos_basis(
                        lambda v: gp_exact.K_noise @ v,
                        Q_resid,
                        target_J=Q_love.shape[1]+Q_resid.shape[1],
                        tol=1e-12,
                        reorthogonalize=True,
                    )

                proj_qr = projection_residual(D_qr, k)
                proj_resid = projection_residual(Q_resid, k)
                proj_ext = projection_residual(Q_ext, k)
                proj_love = projection_residual(Q_love, k)

                for jitter in jitters:
                    try:
                        qr_corr = cg_qr_correction(D_qr, KD_qr, k, jitter=jitter)
                        qr_cov = K_test - qr_corr
                        qr_cov = 0.5 * (qr_cov + qr_cov.T)

                        resid_corr = love_correction(Q_resid, T_resid, k, jitter=jitter)
                        resid_cov = K_test - resid_corr
                        resid_cov = 0.5 * (resid_cov + resid_cov.T)

                        ext_corr = love_correction(Q_ext, T_ext, k, jitter=jitter)
                        ext_cov = K_test - ext_corr
                        ext_cov = 0.5 * (ext_cov + ext_cov.T)

                        love_corr = love_correction(Q_love, T_love, k, jitter=jitter)
                        love_cov = K_test - love_corr
                        love_cov = 0.5 * (love_cov + love_cov.T)

                        qr_var = np.diag(qr_cov)
                        resid_var = np.diag(resid_cov)
                        ext_var = np.diag(ext_cov)
                        love_var = np.diag(love_cov)

                        print(
                            f"{lengthscale:<10.1g} "
                            f"{noise:<10.1e} "
                            f"{jitter:<9.1e} "
                            f"{D_qr.shape[1]:<5d} "
                            f"{Q_resid.shape[1]:<7d} "
                            f"{Q_ext.shape[1]:<6d} "
                            f"{Q_love.shape[1]:<7d} "
                            f"{relative_error(qr_cov, exact_cov):<12.3e} "
                            f"{relative_error(resid_cov, exact_cov):<15.3e} "
                            f"{relative_error(ext_cov, exact_cov):<13.3e} "
                            f"{relative_error(love_cov, exact_cov):<14.3e} "
                            f"{relative_error(qr_var, exact_var):<13.3e} "
                            f"{relative_error(resid_var, exact_var):<16.3e} "
                            f"{relative_error(ext_var, exact_var):<14.3e} "
                            f"{relative_error(love_var, exact_var):<15.3e} "
                            f"{proj_qr:<12.3e} "
                            f"{proj_resid:<12.3e} "
                            f"{proj_ext:<12.3e} "
                            f"{proj_love:<.3e}"
                        )

                    except Exception as e:
                        print(
                            f"{lengthscale:<10.1g} "
                            f"{noise:<10.1e} "
                            f"{jitter:<9.1e} "
                            f"{D_qr.shape[1]:<5d} "
                            f"{Q_resid.shape[1]:<7d} "
                            f"{Q_ext.shape[1]:<6d} "
                            f"{Q_love.shape[1]:<7d} "
                            f"FAILED: {type(e).__name__}: {e}"
                        )

            except Exception as e:
                print(
                    f"{lengthscale:<10.1g} "
                    f"{noise:<10.1e} "
                    f"FAILED during setup: {type(e).__name__}: {e}"
                )


if __name__ == "__main__":
    run_residual_lanczos_comparison()



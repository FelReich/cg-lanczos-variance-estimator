import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.corrections import cg_qr_correction, exact_correction, love_correction
from src.diagnostics import projection_residual, relative_error
from src.gp import GP
from src.kernels import RBFKernel
from src.linalg import cg, lanczos_tridiagonalization, ReorthogonalizationRule
from src.means import ZeroMean


def compare_qr_motivation(
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
    view: str = "both",
) -> None:
    """Compares CG-QR without basis extension against LOVE.

    This script motivates the Lanczos extension step. CG-QR is stable and often
    efficient when the number of CG directions is comparable to the number of
    LOVE Lanczos vectors. However, when CG terminates much earlier than LOVE,
    the QR correction may use a too small Krylov subspace for accurate variance
    estimation.

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
    :param str view: Output mode. Must be `"accuracy"`, `"time"`, or `"both"`. (Default: `"both"`.)
    """
    view = view.lower()
    if view not in {"accuracy", "time", "both"}:
        raise ValueError("view must be one of 'accuracy', 'time', or 'both'.")

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
    rule = ReorthogonalizationRule("always")

    if view in {"accuracy", "both"}:
        accuracy_header = (
            f"{'lengthscale':<10} "
            f"{'noise':<10} "
            f"{'jitter':<9} "
            f"{'J_qr':<6} "
            f"{'J_love':<7} "
            f"{'rel_qr_exact':<15} "
            f"{'rel_love_exact':<16} "
            f"{'diag_qr_exact':<16} "
            f"{'diag_love_exact':<17} "
            f"{'proj_qr':<12} "
            f"{'proj_love':<12}"
        )
        print(accuracy_header)
        print("-" * len(accuracy_header))

    if view in {"time", "both"}:
        if view == "both":
            print()

        time_header = (
            f"{'lengthscale':<10} "
            f"{'noise':<10} "
            f"{'jitter':<9} "
            f"{'J_qr':<6} "
            f"{'J_love':<7} "
            f"{'time_qr_fit':<13} "
            f"{'time_qr_corr':<13} "
            f"{'time_love_cg_fit':<17} "
            f"{'time_love_fit':<14} "
            f"{'time_love_corr':<15} "
            f"{'rel_diff_qr_love':<18}"
        )
        print(time_header)
        print("-" * len(time_header))

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

                t0 = time.perf_counter()
                _, D_qr, KD_qr = cg(
                    lambda v: gp.K_noise @ v,
                    gp.centered_y,
                    J=cg_J,
                    tol=1e-6,
                    save_directions=True,
                    reorthogonalization_rule=rule,
                )
                t1 = time.perf_counter()

                t2 = time.perf_counter()
                _ = cg(
                    lambda v: gp.K_noise @ v,
                    gp.centered_y,
                    J=cg_J,
                    tol=1e-6,
                    save_directions=False,
                    reorthogonalization_rule=None,
                )
                t3 = time.perf_counter()

                t4 = time.perf_counter()
                Q_love, T_love = lanczos_tridiagonalization(
                    lambda v: gp.K_noise @ v,
                    gp.centered_y,
                    num_iter=lanczos_J,
                    tol=1e-6,
                    reorthogonalize=True,
                )
                t5 = time.perf_counter()

                proj_qr = projection_residual(D_qr, k)
                proj_love = projection_residual(Q_love, k)

                time_qr_fit = t1 - t0
                time_love_cg_fit = t3 - t2
                time_love_fit = t5 - t4

                for jitter in jitters:
                    try:
                        t6 = time.perf_counter()
                        qr_corr = cg_qr_correction(D_qr, KD_qr, k, jitter=jitter)
                        t7 = time.perf_counter()

                        love_corr = love_correction(Q_love, T_love, k, jitter=jitter)
                        t8 = time.perf_counter()

                        qr_cov = K_test - qr_corr
                        love_cov = K_test - love_corr

                        qr_cov = 0.5 * (qr_cov + qr_cov.T)
                        love_cov = 0.5 * (love_cov + love_cov.T)

                        qr_var = np.diag(qr_cov)
                        love_var = np.diag(love_cov)

                        time_qr_corr = t7 - t6
                        time_love_corr = t8 - t7

                        total_qr = time_qr_fit + time_qr_corr
                        total_love = time_love_cg_fit + time_love_fit + time_love_corr
                        rel_diff_qr_love = (total_qr - total_love) / total_love

                        if view in {"accuracy", "both"}:
                            print(
                                f"{lengthscale:<10.1g} "
                                f"{noise:<10.1e} "
                                f"{jitter:<9.1e} "
                                f"{D_qr.shape[1]:<6d} "
                                f"{Q_love.shape[1]:<7d} "
                                f"{relative_error(qr_cov, exact_cov):<15.3e} "
                                f"{relative_error(love_cov, exact_cov):<16.3e} "
                                f"{relative_error(qr_var, exact_var):<16.3e} "
                                f"{relative_error(love_var, exact_var):<17.3e} "
                                f"{proj_qr:<12.3e} "
                                f"{proj_love:<12.3e}"
                            )

                        if view in {"time", "both"}:
                            print(
                                f"{lengthscale:<10.1g} "
                                f"{noise:<10.1e} "
                                f"{jitter:<9.1e} "
                                f"{D_qr.shape[1]:<6d} "
                                f"{Q_love.shape[1]:<7d} "
                                f"{time_qr_fit:<13.3e} "
                                f"{time_qr_corr:<13.3e} "
                                f"{time_love_cg_fit:<17.3e} "
                                f"{time_love_fit:<14.3e} "
                                f"{time_love_corr:<15.3e} "
                                f"{rel_diff_qr_love:<18.3e}"
                            )

                    except Exception as e:
                        print(
                            f"{lengthscale:<10.1g} "
                            f"{noise:<10.1e} "
                            f"{jitter:<9.1e} "
                            f"FAILED: {type(e).__name__}: {e}"
                        )

            except Exception as e:
                print(
                    f"{lengthscale:<10.1g} "
                    f"{noise:<10.1e} "
                    f"FAILED during setup: {type(e).__name__}: {e}"
                )


if __name__ == "__main__":
    compare_qr_motivation(view="time")
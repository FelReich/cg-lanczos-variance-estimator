import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.corrections import exact_correction, love_correction
from src.diagnostics import projection_residual, relative_error
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


def compare_lanczos_extended_love(
    *,
    n: int = 10000,
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
    view: str = "both",
) -> None:
    """Compares CG-based Lanczos extension against LOVE.

    LOVE uses a plain CG solve for the predictive mean and a separate Lanczos
    decomposition for the covariance correction. The extended method uses CG with
    reorthogonalization to recover an initial basis, then extends this basis with
    additional Lanczos steps.

    :param int n: Number of training points. (Default: `1000`.)
    :param int m: Number of test points. (Default: `100`.)
    :param int cg_J: Maximum number of CG iterations. (Default: `100`.)
    :param int lanczos_J: Maximum number of Lanczos iterations for LOVE. (Default: `100`.)
    :param float outputscale: Kernel outputscale. (Default: `1.0`.)
    :param list lengthscales: Lengthscales used in the experiment.
    :param list noises: Noise levels used in the experiment.
    :param list jitters: Jitter values used in approximate covariance corrections.
    :param tuple domain: Interval from which training and test inputs are sampled. (Default: `(-3.0, 3.0)`.)
    :param int seed: Random seed. (Default: `123`.)
    :param float extension_tol: Breakdown tolerance for basis extension. (Default: `1e-14`.)
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

    if view in {"accuracy", "both"}:
        accuracy_header = (
            "lengthscale noise      jitter    "
            "J_resid J_ext J_love "
            "rel_ext_exact rel_love_exact "
            "diag_ext_exact diag_love_exact "
            "proj_ext      proj_love"
        )
        print(accuracy_header)
        print("-" * len(accuracy_header))

    if view in {"time", "both"}:
        if view == "both":
            print()

        time_header = (
            "lengthscale noise      jitter    "
            "J_resid J_ext J_love "
            "time_resid_fit time_extend time_ext_corr "
            "time_love_cg_fit time_love_fit time_love_corr "
            "rel_diff_extend_love"
        )
        print(time_header)
        print("-" * len(time_header))

    for lengthscale in lengthscales:
        kernel = RBFKernel(lengthscale=lengthscale, outputscale=outputscale)

        for noise in noises:
            try:
                gp_exact = GP(X_train, y_train, kernel=kernel, mean=mean, noise=noise)
                gp_exact.compute_posterior(method="exact")

                K_test = gp_exact.prior_covariance(X_test)
                k = gp_exact.train_test_covariance(X_test)

                exact_corr = exact_correction(gp_exact.K_noise, k)
                exact_cov = K_test - exact_corr
                exact_cov = 0.5 * (exact_cov + exact_cov.T)
                exact_var = np.diag(exact_cov)

                rule = ReorthogonalizationRule(mode="always")

                t0 = time.perf_counter()
                _, Q_resid, KQ_resid = cg_store_lanczos_basis(
                    lambda v: gp_exact.K_noise @ v,
                    gp_exact.centered_y,
                    J=cg_J,
                    tol=1e-6,
                    reorthogonalization_rule=rule,
                )
                t1 = time.perf_counter()

                t2 = time.perf_counter()
                _ = cg(
                    lambda v: gp_exact.K_noise @ v,
                    gp_exact.centered_y,
                    J=cg_J,
                    tol=1e-6,
                    save_directions=False,
                    reorthogonalization_rule=None,
                )
                t3 = time.perf_counter()

                t4 = time.perf_counter()
                Q_love, T_love = lanczos_tridiagonalization(
                    lambda v: gp_exact.K_noise @ v,
                    gp_exact.centered_y,
                    num_iter=lanczos_J,
                    tol=1e-6,
                    reorthogonalize=True,
                )
                t5 = time.perf_counter()

                target_J = max(Q_resid.shape[1], Q_love.shape[1])

                t6 = time.perf_counter()
                Q_ext, T_ext = extend_lanczos_basis(
                    lambda v: gp_exact.K_noise @ v,
                    Q_resid,
                    KQ_resid,
                    target_J=target_J,
                    tol=extension_tol,
                )
                t7 = time.perf_counter()

                proj_ext = projection_residual(Q_ext, k)
                proj_love = projection_residual(Q_love, k)

                time_resid_fit = t1 - t0
                time_love_cg_fit = t3 - t2
                time_love_fit = t5 - t4
                time_extend = t7 - t6

                for jitter in jitters:
                    try:
                        t8 = time.perf_counter()
                        ext_corr = love_correction(Q_ext, T_ext, k, jitter=jitter)
                        t9 = time.perf_counter()

                        love_corr = love_correction(Q_love, T_love, k, jitter=jitter)
                        t10 = time.perf_counter()

                        ext_cov = K_test - ext_corr
                        love_cov = K_test - love_corr

                        ext_cov = 0.5 * (ext_cov + ext_cov.T)
                        love_cov = 0.5 * (love_cov + love_cov.T)

                        ext_var = np.diag(ext_cov)
                        love_var = np.diag(love_cov)

                        time_ext_corr = t9 - t8
                        time_love_corr = t10 - t9

                        total_extend = time_resid_fit + time_extend + time_ext_corr
                        total_love = time_love_cg_fit + time_love_fit + time_love_corr

                        rel_diff_extend_love = (total_extend - total_love) / total_love

                        if view in {"accuracy", "both"}:
                            print(
                                f"{lengthscale:<10.1g} "
                                f"{noise:<10.1e} "
                                f"{jitter:<9.1e} "
                                f"{Q_resid.shape[1]:<7d} "
                                f"{Q_ext.shape[1]:<5d} "
                                f"{Q_love.shape[1]:<7d} "
                                f"{relative_error(ext_cov, exact_cov):<13.3e} "
                                f"{relative_error(love_cov, exact_cov):<14.3e} "
                                f"{relative_error(ext_var, exact_var):<14.3e} "
                                f"{relative_error(love_var, exact_var):<15.3e} "
                                f"{proj_ext:<12.3e} "
                                f"{proj_love:<.3e}"
                            )

                        if view in {"time", "both"}:
                            print(
                                f"{lengthscale:<10.1g} "
                                f"{noise:<10.1e} "
                                f"{jitter:<9.1e} "
                                f"{Q_resid.shape[1]:<7d} "
                                f"{Q_ext.shape[1]:<5d} "
                                f"{Q_love.shape[1]:<7d} "
                                f"{time_resid_fit:<14.3e} "
                                f"{time_extend:<11.3e} "
                                f"{time_ext_corr:<13.3e} "
                                f"{time_love_cg_fit:<16.3e} "
                                f"{time_love_fit:<13.3e} "
                                f"{time_love_corr:<14.3e} "
                                f"{rel_diff_extend_love:<.3e}"
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
    compare_lanczos_extended_love(view="time")
import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diagnostics import projection_residual, relative_error
from src.gp import GP
from src.kernels import RBFKernel
from src.linalg import ReorthogonalizationRule
from src.means import ZeroMean
from src.diagnostics import conjugacy_error


def run_variance_comparison(
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
    cg_rayleigh_tol: float = 1,
    cg_norm_tol: float = 1,
    lanczos_reorthogonalize: bool = True,
    output: str = "both",
):
    """Runs accuracy and runtime comparisons for CG-Cholesky, CG-QR, and LOVE.

    The exact posterior covariance is used as the reference. The CG methods use
    stored CG search directions, while LOVE uses a separate Lanczos basis for
    the covariance correction.

    :param int seed: Random seed. (Default: `123`.)
    :param int n: Number of training points. (Default: `1000`.)
    :param int m: Number of test points. (Default: `100`.)
    :param tuple domain: Interval from which training and test inputs are sampled. (Default: `(-3.0, 3.0)`.)
    :param int cg_J: Maximum number of CG iterations. (Default: `100`.)
    :param int lanczos_J: Maximum number of Lanczos iterations for LOVE. (Default: `100`.)
    :param float outputscale: Kernel outputscale. (Default: `1.0`.)
    :param list lengthscales: Lengthscales used in the experiment.
    :param list noises: Noise levels used in the experiment.
    :param list jitters: Jitter values used in approximate covariance corrections.
    :param str cg_reorthogonalization_mode: Reorthogonalization mode for CG search directions. (Default: `"always"`.)
    :param int cg_reorthogonalization_every: Frequency used when `cg_reorthogonalization_mode="every"`. (Default: `1`.)
    :param int cg_reorthogonalization_start: First iteration at which CG reorthogonalization is allowed. (Default: `1`.)
    :param float cg_rayleigh_tol: Rayleigh quotient threshold used when `cg_reorthogonalization_mode="rayleigh"`.
    :param float cg_norm_tol: Direction norm threshold used when `cg_reorthogonalization_mode="norm"`.
    :param bool lanczos_reorthogonalize: If True, reorthogonalize Lanczos vectors in LOVE. (Default: True.)
    :param str output: Output mode. Must be `"accuracy"`, `"time"`, or `"both"`. (Default: `"both"`.)
    """
    if lengthscales is None:
        lengthscales = [0.1, 0.3, 1.0, 3.0, 10.0]

    if noises is None:
        noises = [1e-6, 1e-4, 1e-2, 1.0]

    if jitters is None:
        jitters = [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]

    output = output.lower()
    if output not in {"accuracy", "time", "both"}:
        raise ValueError("output must be one of 'accuracy', 'time', or 'both'.")

    cg_reorthogonalization_rule = ReorthogonalizationRule(
        mode=cg_reorthogonalization_mode,
        every=cg_reorthogonalization_every,
        start=cg_reorthogonalization_start,
        rayleigh_tol=cg_rayleigh_tol,
        norm_tol=cg_norm_tol,
    )

    rng = np.random.default_rng(seed)

    X_train = rng.uniform(domain[0], domain[1], (n, 1))
    X_test = rng.uniform(domain[0], domain[1], (m, 1))

    def f(x):
        return np.sin(x) + np.cos(3.0 * x) + np.exp(-x**2)

    y_train = f(X_train).reshape(-1)
    mean = ZeroMean()

    base_header = (
        "lengthscale noise      jitter    "
        "J_cg  J_love "
    )

    accuracy_header = (
        "rel_chol_exact rel_qr_exact   rel_love_exact "
        "rel_chol_love  rel_qr_love   "
        "diag_chol_exact diag_qr_exact diag_love_exact "
        "proj_resid   conj_D        cond_G        "
    )

    time_header = (
        "time_cg_fit time_chol time_qr time_love_fit time_love"
    )

    if output == "accuracy":
        header = base_header + accuracy_header
    elif output == "time":
        header = base_header + time_header
    else:
        header = base_header + accuracy_header + time_header

    print(header)
    print("-" * len(header))

    for lengthscale in lengthscales:
        kernel = RBFKernel(lengthscale=lengthscale, outputscale=outputscale)

        for noise in noises:
            try:
                gp_exact = GP(
                    X_train,
                    y_train,
                    kernel=kernel,
                    mean=mean,
                    noise=noise,
                )
                gp_exact.compute_posterior(method="exact")
                exact_cov = gp_exact.predict_covariance(X_test)
                exact_var = np.diag(exact_cov)

                gp_cg = GP(
                    X_train,
                    y_train,
                    kernel=kernel,
                    mean=mean,
                    noise=noise,
                )

                t0 = time.time()
                gp_cg.compute_posterior(
                    method="cg",
                    cg_J=cg_J,
                    cg_reorthogonalization_rule=cg_reorthogonalization_rule,
                )
                t1 = time.time()

                D = gp_cg.D
                KD = gp_cg.KD
                k = gp_cg.train_test_covariance(X_test)

                proj_resid = projection_residual(D, k)
                conj_D = conjugacy_error(D, KD)
                cond_G = np.linalg.cond(D.T @ KD)

                gp_love = GP(
                    X_train,
                    y_train,
                    kernel=kernel,
                    mean=mean,
                    noise=noise,
                )

                t_love_fit0 = time.time()
                gp_love.compute_posterior(
                    method="love",
                    cg_J=cg_J,
                    lanczos_J=lanczos_J,
                    lanczos_reorthogonalize=lanczos_reorthogonalize,
                )
                t_love_fit1 = time.time()

                for jitter in jitters:
                    gp_cg.jitter = jitter
                    gp_love.jitter = jitter

                    try:
                        t2 = time.time()
                        chol_cov = gp_cg.predict_covariance(
                            X_test,
                            cg_correction_method="cholesky",
                        )
                        t3 = time.time()

                        qr_cov = gp_cg.predict_covariance(
                            X_test,
                            cg_correction_method="qr",
                        )
                        t4 = time.time()

                        love_cov = gp_love.predict_covariance(X_test)
                        t5 = time.time()

                        chol_var = np.diag(chol_cov)
                        qr_var = np.diag(qr_cov)
                        love_var = np.diag(love_cov)

                        rel_chol_exact = relative_error(chol_cov, exact_cov)
                        rel_qr_exact = relative_error(qr_cov, exact_cov)
                        rel_love_exact = relative_error(love_cov, exact_cov)

                        rel_chol_love = relative_error(chol_cov, love_cov)
                        rel_qr_love = relative_error(qr_cov, love_cov)

                        diag_chol_exact = relative_error(chol_var, exact_var)
                        diag_qr_exact = relative_error(qr_var, exact_var)
                        diag_love_exact = relative_error(love_var, exact_var)

                        base_row = (
                            f"{lengthscale:<10.1g} "
                            f"{noise:<10.1e} "
                            f"{jitter:<9.1e} "
                            f"{D.shape[1]:<5d} "
                            f"{gp_love.Q.shape[1]:<7d} "
                        )

                        accuracy_row = (
                            f"{rel_chol_exact:<15.3e} "
                            f"{rel_qr_exact:<14.3e} "
                            f"{rel_love_exact:<15.3e} "
                            f"{rel_chol_love:<15.3e} "
                            f"{rel_qr_love:<13.3e} "
                            f"{diag_chol_exact:<16.3e} "
                            f"{diag_qr_exact:<13.3e} "
                            f"{diag_love_exact:<15.3e} "
                            f"{proj_resid:<13.3e} "
                            f"{conj_D:<13.3e} "
                            f"{cond_G:<13.3e} "
                        )

                        time_row = (
                            f"{t1 - t0:<11.3e} "
                            f"{t3 - t2:<9.3e} "
                            f"{t4 - t3:<7.3e} "
                            f"{t_love_fit1 - t_love_fit0:<13.3e} "
                            f"{t5 - t4:<.3e}"
                        )

                        if output == "accuracy":
                            print(base_row + accuracy_row)
                        elif output == "time":
                            print(base_row + time_row)
                        else:
                            print(base_row + accuracy_row + time_row)

                    except Exception as e:
                        print(
                            f"{lengthscale:<10.1g} "
                            f"{noise:<10.1e} "
                            f"{jitter:<9.1e} "
                            f"{D.shape[1]:<5d} "
                            f"{gp_love.Q.shape[1]:<7d} "
                            f"FAILED: {type(e).__name__}: {e}"
                        )

            except Exception as e:
                print(
                    f"{lengthscale:<10.1g} "
                    f"{noise:<10.1e} "
                    f"FAILED during setup: {type(e).__name__}: {e}"
                )


if __name__ == "__main__":
    run_variance_comparison(output="accuracy")
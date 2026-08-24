import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diagnostics import projection_residual, relative_error
from src.gp import GP
from src.kernels import RBFKernel
from src.linalg import ReorthogonalizationRule, cg
from src.means import ZeroMean


def recursive_naive_correction(matmul, D: np.ndarray, k: np.ndarray, jitter: float = 0.0) -> np.ndarray:
    D = np.asarray(D, dtype=float)
    k = np.asarray(k, dtype=float)

    n, J = D.shape
    S = np.zeros((n, J))

    for j in range(J):
        action = D[:, j]
        A_action = np.asarray(matmul(action), dtype=float).reshape(-1)

        if j > 0:
            S_prev = S[:, :j]
            conjugate_action = action - S_prev @ (S_prev.T @ A_action)
        else:
            conjugate_action = action.copy()

        A_conjugate_action = np.asarray(matmul(conjugate_action), dtype=float).reshape(-1)
        weight = np.dot(action, A_conjugate_action) + jitter

        if weight <= 0:
            raise np.linalg.LinAlgError("Non-positive weight in recursive naive correction.")

        S[:, j] = conjugate_action / np.sqrt(weight)

    Z = S.T @ k
    return Z.T @ Z


def make_data(seed: int, n: int, m: int, domain: tuple[float, float]):
    rng = np.random.default_rng(seed)

    X_train = rng.uniform(domain[0], domain[1], (n, 1))
    X_test = np.linspace(domain[0], domain[1], m).reshape(-1, 1)

    def f(x):
        return np.sin(x) + np.cos(3.0 * x) + np.exp(-x**2)

    y_train = f(X_train).reshape(-1)
    true_y = f(X_test).reshape(-1)

    return X_train, y_train, X_test, true_y


def run_naive_vs_love_comparison(
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
):
    if lengthscales is None:
        lengthscales = [0.1, 0.3, 1.0, 3.0, 10.0]

    if noises is None:
        noises = [1e-6, 1e-4, 1e-2, 1.0]

    if jitters is None:
        jitters = [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]

    X_train, y_train, X_test, _ = make_data(seed, n, m, domain)
    mean = ZeroMean()

    header = (
        "lengthscale noise      jitter    "
        "J_raw J_love "
        "rel_naive_exact rel_love_exact rel_naive_love "
        "diag_naive_exact diag_love_exact "
        "proj_raw     "
        "time_raw_cg time_naive time_love_fit time_love"
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

                t0 = time.time()
                _, D_raw, _ = cg(
                    lambda v: gp_exact.K_noise @ v,
                    gp_exact.centered_y,
                    J=cg_J,
                    tol=0.0,
                    save_directions=True,
                    reorthogonalization_rule=ReorthogonalizationRule(mode="never"),
                )
                t1 = time.time()

                gp_love = GP(X_train, y_train, kernel=kernel, mean=mean, noise=noise)

                t2 = time.time()
                gp_love.compute_posterior(
                    method="love",
                    cg_J=cg_J,
                    lanczos_J=lanczos_J,
                    tol=0.0,
                    lanczos_reorthogonalize=True,
                )
                t3 = time.time()

                proj_raw = projection_residual(D_raw, k)

                for jitter in jitters:
                    try:
                        t4 = time.time()
                        naive_corr = recursive_naive_correction(
                            lambda v: gp_exact.K_noise @ v,
                            D_raw,
                            k,
                            jitter=jitter,
                        )
                        naive_cov = K_test - naive_corr
                        naive_cov = 0.5 * (naive_cov + naive_cov.T)
                        t5 = time.time()

                        gp_love.jitter = jitter
                        love_cov = gp_love.predict_covariance(X_test)
                        t6 = time.time()

                        naive_var = np.diag(naive_cov)
                        love_var = np.diag(love_cov)

                        print(
                            f"{lengthscale:<10.1g} "
                            f"{noise:<10.1e} "
                            f"{jitter:<9.1e} "
                            f"{D_raw.shape[1]:<5d} "
                            f"{gp_love.Q.shape[1]:<7d} "
                            f"{relative_error(naive_cov, exact_cov):<15.3e} "
                            f"{relative_error(love_cov, exact_cov):<14.3e} "
                            f"{relative_error(naive_cov, love_cov):<15.3e} "
                            f"{relative_error(naive_var, exact_var):<16.3e} "
                            f"{relative_error(love_var, exact_var):<15.3e} "
                            f"{proj_raw:<12.3e} "
                            f"{t1 - t0:<11.3e} "
                            f"{t5 - t4:<10.3e} "
                            f"{t3 - t2:<13.3e} "
                            f"{t6 - t5:<.3e}"
                        )

                    except Exception as e:
                        print(
                            f"{lengthscale:<10.1g} "
                            f"{noise:<10.1e} "
                            f"{jitter:<9.1e} "
                            f"{D_raw.shape[1]:<5d} "
                            f"{gp_love.Q.shape[1]:<7d} "
                            f"FAILED: {type(e).__name__}: {e}"
                        )

            except Exception as e:
                print(
                    f"{lengthscale:<10.1g} "
                    f"{noise:<10.1e} "
                    f"FAILED during setup: {type(e).__name__}: {e}"
                )


def plot_naive_vs_love_comparison(
    *,
    seed: int = 123,
    n: int = 1000,
    m: int = 1000,
    domain: tuple[float, float] = (-3.0, 3.0),
    cg_J: int = 100,
    lanczos_J: int = 100,
    lengthscale: float = 1.0,
    outputscale: float = 1.0,
    noise: float = 1.0,
    jitter: float = 1e-6,
):
    X_train, y_train, X_test, true_y = make_data(seed, n, m, domain)

    kernel = RBFKernel(lengthscale=lengthscale, outputscale=outputscale)
    mean = ZeroMean()

    gp_exact = GP(X_train, y_train, kernel=kernel, mean=mean, noise=noise)
    gp_exact.compute_posterior(method="exact")

    pred_mean = gp_exact.predict_mean(X_test)
    exact_cov = gp_exact.predict_covariance(X_test)
    exact_var = np.maximum(np.diag(exact_cov), 0.0)
    exact_std = np.sqrt(exact_var)

    K_test = gp_exact.prior_covariance(X_test)
    k = gp_exact.train_test_covariance(X_test)

    _, D_raw, _ = cg(
        lambda v: gp_exact.K_noise @ v,
        gp_exact.centered_y,
        J=cg_J,
        tol=0.0,
        save_directions=True,
        reorthogonalization_rule=ReorthogonalizationRule(mode="never"),
    )

    naive_corr = recursive_naive_correction(
        lambda v: gp_exact.K_noise @ v,
        D_raw,
        k,
        jitter=jitter,
    )
    naive_cov = K_test - naive_corr
    naive_cov = 0.5 * (naive_cov + naive_cov.T)
    naive_var = np.maximum(np.diag(naive_cov), 0.0)

    gp_love = GP(X_train, y_train, kernel=kernel, mean=mean, noise=noise)
    gp_love.compute_posterior(
        method="love",
        cg_J=cg_J,
        lanczos_J=lanczos_J,
        tol=0.0,
        jitter=jitter,
        lanczos_reorthogonalize=True,
    )
    love_var = np.maximum(gp_love.predict_variance(X_test), 0.0)

    methods = [
        ("Naive recursive", naive_var, D_raw.shape[1]),
        ("LOVE", love_var, gp_love.Q.shape[1]),
    ]

    x = X_test.squeeze()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, (title, variance, J_eff) in zip(axes, methods):
        std = np.sqrt(variance)
        lower = pred_mean - 2.0 * std
        upper = pred_mean + 2.0 * std

        exact_lower = pred_mean - 2.0 * exact_std
        exact_upper = pred_mean + 2.0 * exact_std

        ax.plot(x, true_y, "k--", lw=2, label="true function")
        ax.scatter(X_train.squeeze(), y_train, s=10, color="black", alpha=0.25, label="training data")
        ax.plot(x, pred_mean, color="black", lw=2, label="predictive mean")

        ax.plot(x, exact_lower, color="red", linestyle="--", lw=2, label="exact boundary")
        ax.plot(x, exact_upper, color="red", linestyle="--", lw=2)

        ax.fill_between(x, lower, upper, color="tab:orange", alpha=0.25, label="approx. area")

        ax.set_title(f"{title}, J={J_eff}")
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("y")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5)

    fig.suptitle(
        f"Naive vs LOVE: lengthscale={lengthscale}, noise={noise}, jitter={jitter}",
        y=1.05,
    )
    fig.tight_layout()
    plt.show()

    print("Naive variance min/max:", np.min(naive_var), np.max(naive_var))
    print("LOVE variance min/max:", np.min(love_var), np.max(love_var))
    print("Exact variance min/max:", np.min(exact_var), np.max(exact_var))
    print("Projection residual raw CG:", projection_residual(D_raw, k))


if __name__ == "__main__":
    run_naive_vs_love_comparison()
    #plot_naive_vs_love_comparison()
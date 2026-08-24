import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diagnostics import projection_residual, relative_error
from src.gp import GP
from src.kernels import RBFKernel
from src.linalg import ReorthogonalizationRule, cg
from src.linalg.cg import cg_with_basis_extension
from src.means import ZeroMean
from src.corrections import cg_qr_correction


def make_data(seed: int, n: int, m: int, domain: tuple[float, float]):
    rng = np.random.default_rng(seed)

    X_train = rng.uniform(domain[0], domain[1], (n, 1))
    X_test = rng.uniform(domain[0], domain[1], (m, 1))

    def f(x):
        return np.sin(x) + np.cos(3.0 * x) + np.exp(-x**2)

    y_train = f(X_train).reshape(-1)

    return X_train, y_train, X_test

def cg_qr_correction_rank_truncated(
    D: np.ndarray,
    KD: np.ndarray,
    k: np.ndarray,
    jitter: float = 0.0,
    rank_tol: float = 1e-10,
) -> np.ndarray:
    Q, R = np.linalg.qr(D, mode="reduced")

    diag_R = np.abs(np.diag(R))
    scale = max(np.max(diag_R), 1.0)
    rank = int(np.sum(diag_R > rank_tol * scale))

    if rank == 0:
        return np.zeros((k.shape[1], k.shape[1]))

    Q = Q[:, :rank]
    R = R[:rank, :rank]
    KD = KD[:, :rank]

    T = np.linalg.solve(R.T, (Q.T @ KD).T).T
    T = 0.5 * (T + T.T)

    L = np.linalg.cholesky(T + jitter * np.eye(T.shape[0]))
    Z = np.linalg.solve(L, Q.T @ k)

    return Z.T @ Z

def run_cg_repair_comparison(
    *,
    seed: int = 123,
    n: int = 1000,
    m: int = 100,
    domain: tuple[float, float] = (-3.0, 3.0),
    cg_J: int = 100,
    repair_min_iter: int = 100,
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
    repair_tol: float = 1e-12,
    denominator_tol: float = 1e-14,
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
        "J_cg  J_repair J_love "
        "rel_cg_exact rel_repair_exact rel_love_exact "
        "diag_cg_exact diag_repair_exact diag_love_exact "
        "proj_cg     proj_repair "
        "time_cg_fit time_repair_fit time_cg_qr time_repair_qr time_love_fit time_love"
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
                _, D_cg, KD_cg = cg(
                    lambda v: gp_exact.K_noise @ v,
                    gp_exact.centered_y,
                    J=cg_J,
                    tol=1e-6,
                    save_directions=True,
                    reorthogonalization_rule=rule,
                )
                t1 = time.time()

                gp_love = GP(X_train, y_train, kernel=kernel, mean=mean, noise=noise)

                t2 = time.time()
                gp_love.compute_posterior(
                    method="love",
                    cg_J=cg_J,
                    lanczos_J=lanczos_J,
                    tol=1e-6,
                    lanczos_reorthogonalize=True,
                )
                t3 = time.time()

                repair_target_iter = max(D_cg.shape[1], gp_love.Q.shape[1])

                t4 = time.time()
                _, D_repair, KD_repair = cg_with_basis_extension(
                    lambda v: gp_exact.K_noise @ v,
                    gp_exact.centered_y,
                    J=cg_J,
                    tol=1e-6,
                    save_directions=True,
                    reorthogonalization_rule=rule,
                    min_iter=repair_target_iter,
                    repair_tol=repair_tol,
                    denominator_tol=denominator_tol,
                )
                t5 = time.time()

                proj_cg = projection_residual(D_cg, k)
                proj_repair = projection_residual(D_repair, k)

                for jitter in jitters:
                    try:
                        t6 = time.time()
                        cg_corr = cg_qr_correction(D_cg, KD_cg, k, jitter=jitter)
                        cg_cov = K_test - cg_corr
                        cg_cov = 0.5 * (cg_cov + cg_cov.T)
                        t7 = time.time()

                        repair_corr = cg_qr_correction_rank_truncated(
                            D_repair,
                            KD_repair,
                            k,
                            jitter=jitter,
                        )
                        repair_cov = K_test - repair_corr
                        repair_cov = 0.5 * (repair_cov + repair_cov.T)
                        t8 = time.time()

                        gp_love.jitter = jitter
                        love_cov = gp_love.predict_covariance(X_test)
                        t9 = time.time()

                        cg_var = np.diag(cg_cov)
                        repair_var = np.diag(repair_cov)
                        love_var = np.diag(love_cov)

                        print(
                            f"{lengthscale:<10.1g} "
                            f"{noise:<10.1e} "
                            f"{jitter:<9.1e} "
                            f"{D_cg.shape[1]:<5d} "
                            f"{D_repair.shape[1]:<8d} "
                            f"{gp_love.Q.shape[1]:<7d} "
                            f"{relative_error(cg_cov, exact_cov):<13.3e} "
                            f"{relative_error(repair_cov, exact_cov):<16.3e} "
                            f"{relative_error(love_cov, exact_cov):<14.3e} "
                            f"{relative_error(cg_var, exact_var):<13.3e} "
                            f"{relative_error(repair_var, exact_var):<17.3e} "
                            f"{relative_error(love_var, exact_var):<15.3e} "
                            f"{proj_cg:<12.3e} "
                            f"{proj_repair:<12.3e} "
                            f"{t1 - t0:<11.3e} "
                            f"{t5 - t4:<15.3e} "
                            f"{t7 - t6:<10.3e} "
                            f"{t8 - t7:<14.3e} "
                            f"{t3 - t2:<13.3e} "
                            f"{t9 - t8:<.3e}"
                        )

                    except Exception as e:
                        print(
                            f"{lengthscale:<10.1g} "
                            f"{noise:<10.1e} "
                            f"{jitter:<9.1e} "
                            f"{D_cg.shape[1]:<5d} "
                            f"{D_repair.shape[1]:<8d} "
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
    run_cg_repair_comparison()
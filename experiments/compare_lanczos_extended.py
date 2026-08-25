import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.corrections import cg_qr_correction, exact_correction, love_correction
from src.diagnostics import projection_residual, relative_error
from src.gp import GP
from src.kernels import RBFKernel
from src.linalg import ReorthogonalizationRule, cg, lanczos_tridiagonalization
from src.linalg.cg import cg_store_lanczos_basis, cg_store_lanczos_basis_kq, cg_store_lanczos_basis_true
from src.linalg.lanczos import extend_lanczos_basis, extend_lanczos_basis_kq, extend_lanczos_basis_naive
from src.means import ZeroMean


def compare_lanczos_extended(
    *,
    n: int = 1000,
    m: int = 100,
    cg_J: int = 100,
    lanczos_J: int = 100,
    outputscale: float = 1.0,
    lengthscales: list[float] | None = None,
    noises: list[float] | None = None,
    jitters: list[float] | None = None,
    domain: tuple[float, float] = (-100.0, 100.0),
    seed: int = 123,
    view: str = "both",
) -> None:
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
            "J_qr  J_resid J_ext J_love "
            "rel_qr_exact  rel_ext_exact rel_love_exact "
            "diag_qr_exact diag_ext_exact diag_love_exact "
            "proj_qr      proj_ext      proj_love"
        )
        print(accuracy_header)
        print("-" * len(accuracy_header))

    if view in {"time", "both"}:
        time_header = (
            "lengthscale noise      jitter    "
            "J_qr  J_resid J_ext J_love "
            "time_qr_fit time_qr_corr "
            "time_resid_fit time_extend time_ext_corr "
            "time_love_fit time_love_corr"
        )
        if view == "both":
            print()
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

                t0 = time.time()
                _, D_qr, KD_qr = cg(
                    lambda v: gp_exact.K_noise @ v,
                    gp_exact.centered_y,
                    J=cg_J,
                    tol=1e-6,
                    save_directions=True,
                    reorthogonalization_rule=rule,
                )
                t1 = time.time()

                t4 = time.time()
                Q_love, T_love = lanczos_tridiagonalization(
                    lambda v: gp_exact.K_noise @ v,
                    gp_exact.centered_y,
                    num_iter=lanczos_J,
                    tol=1e-6,
                    reorthogonalize=True,
                )
                t5 = time.time()

                t2 = time.time()
                _, Q_resid, KQ_resid = cg_store_lanczos_basis_true(
                    lambda v: gp_exact.K_noise @ v,
                    gp_exact.centered_y,
                    J=min(cg_J, Q_love.shape[1]),
                    tol=1e-6,
                    reorthogonalize=True,
                )
                t3 = time.time()

                """t4 = time.time()
                Q_love, T_love = lanczos_tridiagonalization(
                    lambda v: gp_exact.K_noise @ v,
                    gp_exact.centered_y,
                    num_iter=lanczos_J,
                    tol=1e-6,
                    reorthogonalize=True,
                )
                t5 = time.time()"""

                target_J = Q_resid.shape[1] + Q_love.shape[1]//2

                t6 = time.time()
                Q_ext, T_ext = extend_lanczos_basis_kq(
                    lambda v: gp_exact.K_noise @ v,
                    Q_resid,
                    KQ_resid,
                    target_J=target_J,
                    tol=1e-12,
                )
                t7 = time.time()

                proj_qr = projection_residual(D_qr, k)
                proj_ext = projection_residual(Q_ext, k)
                proj_love = projection_residual(Q_love, k)

                for jitter in jitters:
                    try:
                        t8 = time.time()
                        qr_corr = cg_qr_correction(D_qr, KD_qr, k, jitter=jitter)
                        t9 = time.time()

                        ext_corr = love_correction(Q_ext, T_ext, k, jitter=jitter)
                        t10 = time.time()

                        love_corr = love_correction(Q_love, T_love, k, jitter=jitter)
                        t11 = time.time()

                        qr_cov = K_test - qr_corr
                        ext_cov = K_test - ext_corr
                        love_cov = K_test - love_corr

                        qr_cov = 0.5 * (qr_cov + qr_cov.T)
                        ext_cov = 0.5 * (ext_cov + ext_cov.T)
                        love_cov = 0.5 * (love_cov + love_cov.T)

                        qr_var = np.diag(qr_cov)
                        ext_var = np.diag(ext_cov)
                        love_var = np.diag(love_cov)

                        if view in {"accuracy", "both"}:
                            print(
                                f"{lengthscale:<10.1g} "
                                f"{noise:<10.1e} "
                                f"{jitter:<9.1e} "
                                f"{D_qr.shape[1]:<5d} "
                                f"{Q_resid.shape[1]:<7d} "
                                f"{Q_ext.shape[1]:<5d} "
                                f"{Q_love.shape[1]:<7d} "
                                f"{relative_error(qr_cov, exact_cov):<13.3e} "
                                f"{relative_error(ext_cov, exact_cov):<13.3e} "
                                f"{relative_error(love_cov, exact_cov):<14.3e} "
                                f"{relative_error(qr_var, exact_var):<13.3e} "
                                f"{relative_error(ext_var, exact_var):<14.3e} "
                                f"{relative_error(love_var, exact_var):<15.3e} "
                                f"{proj_qr:<12.3e} "
                                f"{proj_ext:<12.3e} "
                                f"{proj_love:<.3e}"
                            )

                        if view in {"time", "both"}:
                            print(
                                f"{lengthscale:<10.1g} "
                                f"{noise:<10.1e} "
                                f"{jitter:<9.1e} "
                                f"{D_qr.shape[1]:<5d} "
                                f"{Q_resid.shape[1]:<7d} "
                                f"{Q_ext.shape[1]:<5d} "
                                f"{Q_love.shape[1]:<7d} "
                                f"{t1 - t0:<11.3e} "
                                f"{t9 - t8:<12.3e} "
                                f"{t3 - t2:<14.3e} "
                                f"{t7 - t6:<11.3e} "
                                f"{t10 - t9:<13.3e} "
                                f"{t5 - t4:<13.3e} "
                                f"{t11 - t10:<.3e}"
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


def compare_lanczos_extended_(
    *,
    n: int = 1000,
    m: int = 100,
    cg_J: int = 100,
    lanczos_J: int = 100,
    outputscale: float = 1.0,
    lengthscales: list[float] | None = None,
    noises: list[float] | None = None,
    jitters: list[float] | None = None,
    domain: tuple[float, float] = (-100.0, 100.0),
    seed: int = 123,
) -> None:
    if lengthscales is None:
        lengthscales = [0.1, 0.3, 1.0, 3.0, 10.0]

    if noises is None:
        noises = [1e-6, 1e-4, 1e-2, 1.0]

    if jitters is None:
        jitters = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2]

    rng = np.random.default_rng(seed)

    X_train = rng.uniform(domain[0], domain[1], size=(n, 1))
    X_test = rng.uniform(domain[0], domain[1], size=(m, 1))

    def f(x):
        return np.sin(x) + np.cos(3 * x) + np.exp(-x**2)

    y_train = f(X_train).reshape(-1)
    mean = ZeroMean()

    header = (
        "lengthscale noise      jitter    "
        "J_qr  J_resid J_ext J_love "
        "rel_qr_exact  rel_ext_exact rel_love_exact "
        "time_qr_fit time_qr_corr "
        "time_resid_fit time_extend time_ext_corr "
        "time_love_fit time_love_corr"
    )
    print(header)
    print("-" * len(header))

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

                rule = ReorthogonalizationRule(mode="always")

                t0 = time.time()
                _, D_qr, KD_qr = cg(
                    lambda v: gp_exact.K_noise @ v,
                    gp_exact.centered_y,
                    J=cg_J,
                    tol=1e-6,
                    save_directions=True,
                    reorthogonalization_rule=rule,
                )
                t1 = time.time()

                t4 = time.time()
                Q_love, T_love = lanczos_tridiagonalization(
                    lambda v: gp_exact.K_noise @ v,
                    gp_exact.centered_y,
                    num_iter=lanczos_J,
                    tol=1e-6,
                    reorthogonalize=True,
                )
                t5 = time.time()

                t2 = time.time()
                _, Q_resid, KQ_resid = cg_store_lanczos_basis_true(
                    lambda v: gp_exact.K_noise @ v,
                    gp_exact.centered_y,
                    J=min(cg_J, Q_love.shape[1]),
                    tol=1e-6,
                    reorthogonalize=True,
                )
                t3 = time.time()

                target_J = max(Q_resid.shape[1] + Q_love.shape[1]//3, Q_love.shape[1])

                t6 = time.time()
                Q_ext, T_ext = extend_lanczos_basis_kq(
                    lambda v: gp_exact.K_noise @ v,
                    Q_resid,
                    KQ_resid,
                    target_J=target_J,
                    tol=1e-12,
                )
                t7 = time.time()

                for jitter in jitters:
                    try:
                        t8 = time.time()
                        qr_corr = cg_qr_correction(D_qr, KD_qr, k, jitter=jitter)
                        t9 = time.time()

                        ext_corr = love_correction(Q_ext, T_ext, k, jitter=jitter)
                        t10 = time.time()

                        love_corr = love_correction(Q_love, T_love, k, jitter=jitter)
                        t11 = time.time()

                        qr_cov = K_test - qr_corr
                        ext_cov = K_test - ext_corr
                        love_cov = K_test - love_corr

                        qr_cov = 0.5 * (qr_cov + qr_cov.T)
                        ext_cov = 0.5 * (ext_cov + ext_cov.T)
                        love_cov = 0.5 * (love_cov + love_cov.T)

                        print(
                            f"{lengthscale:<10.1g} "
                            f"{noise:<10.1e} "
                            f"{jitter:<9.1e} "
                            f"{D_qr.shape[1]:<5d} "
                            f"{Q_resid.shape[1]:<7d} "
                            f"{Q_ext.shape[1]:<5d} "
                            f"{Q_love.shape[1]:<7d} "
                            f"{relative_error(qr_cov, exact_cov):<13.3e} "
                            f"{relative_error(ext_cov, exact_cov):<13.3e} "
                            f"{relative_error(love_cov, exact_cov):<14.3e} "
                            f"{t1 - t0:<11.3e} "
                            f"{t9 - t8:<12.3e} "
                            f"{t3 - t2:<14.3e} "
                            f"{t7 - t6:<11.3e} "
                            f"{t10 - t9:<13.3e} "
                            f"{t5 - t4:<13.3e} "
                            f"{t11 - t10:<.3e}"
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
    compare_lanczos_extended_()
    #compare_lanczos_extended(view="accuracy")
    """
    domain = (-100.0, 100.0)
    rng = np.random.default_rng(123)
    n = 1000
    m = 100
    cg_J = 100
    lanczos_J = 100

    X_train = rng.uniform(domain[0], domain[1], size=(n, 1))
    X_test = rng.uniform(domain[0], domain[1], size=(m, 1))

    def f(x):
        return np.sin(x) + np.cos(3 * x) + np.exp(-x**2)

    y_train = f(X_train).reshape(-1)
    mean = ZeroMean()
    kernel = RBFKernel(lengthscale=0.1, outputscale=1.0)

    gp_exact = GP(X_train, y_train, kernel=kernel, mean=mean, noise=1e-4)
    gp_exact.compute_posterior(method="exact")

    K_test = gp_exact.prior_covariance(X_test)
    k = gp_exact.train_test_covariance(X_test)
    exact_corr = exact_correction(gp_exact.K_noise, k)
    exact_cov = K_test - exact_corr
    exact_cov = 0.5 * (exact_cov + exact_cov.T)
    exact_var = np.diag(exact_cov)

    rule = ReorthogonalizationRule(mode="always")

    _, D_qr, KD_qr = cg(
        lambda v: gp_exact.K_noise @ v,
        gp_exact.centered_y,
        J=cg_J,
        tol=1e-6,
        save_directions=True,
        reorthogonalization_rule=rule,
    )
    _, Q_resid, KQ_resid = cg_store_lanczos_basis_true(
        lambda v: gp_exact.K_noise @ v,
        gp_exact.centered_y,
           J=cg_J,
        tol=1e-6,
        reorthogonalize=True,
    )
    Q_love, T_love = lanczos_tridiagonalization(
        lambda v: gp_exact.K_noise @ v,
        gp_exact.centered_y,
        num_iter=lanczos_J,
        tol=1e-6,
        reorthogonalize=True,
    )

    target_J = Q_resid.shape[1] + Q_love.shape[1]

    Q_ext, T_ext = extend_lanczos_basis_kq(
        lambda v: gp_exact.K_noise @ v,
        Q_resid,
        KQ_resid,
        target_J=target_J,
        tol=1e-10,
    )

    KQ_direct = gp_exact.K_noise @ Q_ext
    T_direct = Q_ext.T @ KQ_direct
    T_direct = 0.5 * (T_direct + T_direct.T)

    print("J_resid:", Q_resid.shape[1])
    print("J_ext:", Q_ext.shape[1])
    print("J_love:", Q_love.shape[1])

    print("orth err Q_ext:", np.linalg.norm(Q_ext.T @ Q_ext - np.eye(Q_ext.shape[1])))

    # If extend_lanczos_basis_kq also internally uses/stores propagated KQ,
    # compare the T it returns against the true projected matrix.
    print("T_ext symmetry:", np.linalg.norm(T_ext - T_ext.T))
    print("T_ext vs direct:", np.linalg.norm(T_ext - T_direct) / np.linalg.norm(T_direct))

    print("min eig T_ext:", np.min(np.linalg.eigvalsh(T_ext)))
    print("min eig T_direct:", np.min(np.linalg.eigvalsh(T_direct)))

    print("proj_ext:", projection_residual(Q_ext, k))
    print("proj_love:", projection_residual(Q_love, k))

    love_corr = love_correction(Q_love, T_love, k, jitter=1e-6)
    love_cov = K_test - love_corr
    print("love rel cov:", relative_error(love_cov, exact_cov))

    
    
    print(np.allclose(T_ext, T_ext.T))

    print(np.linalg.eigvals(T_ext))
    try:
        love_correction(Q_ext, T_ext, k, jitter=0.0)
        print("Done")
    except:
        print("failed")
    """

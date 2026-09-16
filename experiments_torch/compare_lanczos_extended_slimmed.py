import sys
import time
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore", message="CG terminated.*")
warnings.filterwarnings("ignore", message="Failed to initialize NumPy.*")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src_torch.corrections import exact_correction, love_correction
from src_torch.diagnostics import relative_error
from src_torch.gp import GP
from src_torch.kernels import RBFKernel
from src_torch.linalg.cg import linear_cg, cg_store_lanczos_basis
from src_torch.linalg.lanczos import lanczos_tridiag, extend_lanczos_basis
from src_torch.means import ZeroMean


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def compare_lanczos_extended_love(
    *,
    n: int = 1000,
    m: int = 100,
    cg_J: int = 500,
    lanczos_J: int = 500,
    outputscale: float = 1.0,
    lengthscales: list[float] | None = None,
    noises: list[float] | None = None,
    jitters: list[float] | None = None,
    domain: tuple[float, float] = (-10.0, 10.0),
    seed: int = 123,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> None:
    if lengthscales is None:
        lengthscales = [0.1, 0.3, 1.0, 3.0, 10.0]

    if noises is None:
        noises = [1e-6, 1e-4, 1e-2, 1.0]

    if jitters is None:
        jitters = [1e-6]#[1e-8, 1e-6, 1e-4]

    if dtype == torch.float64:
        tol = 1e-6
    else:
        tol = 1e-3

    device = torch.device(device)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    X_train = domain[0] + (domain[1] - domain[0]) * torch.rand(
        n,
        1,
        generator=generator,
        dtype=dtype,
        device=device,
    )
    X_test = domain[0] + (domain[1] - domain[0]) * torch.rand(
        m,
        1,
        generator=generator,
        dtype=dtype,
        device=device,
    )

    def f(x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x) + torch.cos(3.0 * x) + torch.exp(-(x**2))

    y_train = f(X_train).reshape(-1)
    mean = ZeroMean()

    header = (
        "lengthscale noise      jitter    "
        "J_resid J_ext J_love "
        "total_time_ext total_time_love rel_diff_time "
        "rel_ext_exact rel_love_exact "
        "cg_same_exact lanczos_same_exact compare_same"
    )
    print(header)
    print("-" * len(header))

    with torch.no_grad():
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
                    exact_cov = 0.5 * (exact_cov + exact_cov.transpose(-1, -2))

                    rhs = gp_exact.centered_y.reshape(1, -1, 1)
                    matmul_closure = lambda v: torch.matmul(gp_exact.K_noise, v)

                    _sync_if_needed(device)
                    t0 = time.perf_counter()
                    res_cg, Q_resid, T_resid = cg_store_lanczos_basis(
                        matmul_closure,
                        rhs,
                        tolerance=1e-6,
                        max_iter=cg_J,
                    )
                    _sync_if_needed(device)
                    t1 = time.perf_counter()

                    _sync_if_needed(device)
                    t2 = time.perf_counter()
                    res_lanczos = linear_cg(
                        matmul_closure,
                        rhs,
                        tolerance=1e-6,
                        max_iter=cg_J,
                    )
                    _sync_if_needed(device)
                    t3 = time.perf_counter()

                    _sync_if_needed(device)
                    t4 = time.perf_counter()
                    Q_love, T_love = lanczos_tridiag(
                        matmul_closure,
                        max_iter=lanczos_J,
                        dtype=dtype,
                        device=device,
                        matrix_shape=gp_exact.K_noise.shape,
                        batch_shape=rhs.shape[:-2],
                        init_vecs=rhs,
                        num_init_vecs=1,
                        tol=tol,
                    )
                    _sync_if_needed(device)
                    t5 = time.perf_counter()

                    #target_J = max(Q_resid.shape[-1], Q_love.shape[-1])
                    target_J = lanczos_J

                    _sync_if_needed(device)
                    t6 = time.perf_counter()
                    Q_ext, T_ext = extend_lanczos_basis(
                        matmul_closure,
                        max_iter=target_J,
                        dtype=dtype,
                        device=device,
                        matrix_shape=gp_exact.K_noise.shape,
                        q_mat=Q_resid,
                        t_mat=T_resid,
                        tol=tol,
                    )
                    _sync_if_needed(device)
                    t7 = time.perf_counter()

                    rhs_exact = y_train - gp_exact.mean(gp_exact.X_train)
                    res_exact = torch.linalg.solve(gp_exact.K_noise, rhs_exact)

                    Q_ext_2d = Q_ext.squeeze(0)
                    T_ext_2d = T_ext.squeeze(0)
                    Q_love_2d = Q_love.squeeze(0)
                    T_love_2d = T_love.squeeze(0)

                    time_resid_fit = t1 - t0
                    time_love_cg_fit = t3 - t2
                    time_love_fit = t5 - t4
                    time_extend = t7 - t6

                    for jitter in jitters:
                        try:
                            _sync_if_needed(device)
                            t8 = time.perf_counter()
                            ext_corr = love_correction(Q_ext_2d, T_ext_2d, k, jitter=jitter)
                            _sync_if_needed(device)
                            t9 = time.perf_counter()

                            love_corr = love_correction(Q_love_2d, T_love_2d, k, jitter=jitter)
                            _sync_if_needed(device)
                            t10 = time.perf_counter()

                            ext_cov = K_test - ext_corr
                            love_cov = K_test - love_corr

                            ext_cov = 0.5 * (ext_cov + ext_cov.transpose(-1, -2))
                            love_cov = 0.5 * (love_cov + love_cov.transpose(-1, -2))

                            time_ext_corr = t9 - t8
                            time_love_corr = t10 - t9

                            total_time_ext = time_resid_fit + time_extend + time_ext_corr
                            total_time_love = time_love_cg_fit + time_love_fit + time_love_corr
                            rel_diff_time = (total_time_ext - total_time_love) / total_time_love

                            rel_cg_same_exact = torch.linalg.vector_norm(res_cg - res_exact)/torch.linalg.vector_norm(res_exact)
                            rel_lanczos_same_exact = torch.linalg.vector_norm(res_lanczos - res_exact)/torch.linalg.vector_norm(res_exact)

                            C_cg = torch.matmul(Q_ext_2d, torch.matmul(T_ext_2d, Q_ext_2d.T))
                            C_lanczos = torch.matmul(Q_love_2d, torch.matmul(T_love_2d, Q_love_2d.T))

                            rel_cg_lanczos_same = relative_error(C_cg, C_lanczos)


                            print(
                                f"{lengthscale:<10.1g} "
                                f"{noise:<10.1e} "
                                f"{jitter:<9.1e} "
                                f"{Q_resid.shape[-1]:<7d} "
                                f"{Q_ext.shape[-1]:<5d} "
                                f"{Q_love.shape[-1]:<7d} "
                                f"{total_time_ext:<14.3e} "
                                f"{total_time_love:<15.3e} "
                                f"{rel_diff_time:<13.3e} "
                                f"{relative_error(ext_cov, exact_cov):<13.3e} "
                                f"{relative_error(love_cov, exact_cov):<15.3e} "
                                f"{rel_cg_same_exact:<17.3e} "
                                f"{rel_lanczos_same_exact:<16.3e} "
                                f"{rel_cg_lanczos_same:<.3e}"
                            )
                            """
                            print(
                                f"{lengthscale:<10.1g} "
                                f"{noise:<10.1e} "
                                f"{jitter:<9.1e} "
                                f"{Q_resid.shape[-1]:<7d} "
                                f"{Q_ext.shape[-1]:<5d} "
                                f"{Q_love.shape[-1]:<7d} "
                                f"{total_time_ext:<14.3e} "
                                f"{total_time_love:<15.3e} "
                                f"{rel_diff_time:<13.3e} "
                                f"{relative_error(ext_cov, exact_cov):<13.3e} "
                                f"{relative_error(love_cov, exact_cov):<15.3e} "
                            )"""

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
    compare_lanczos_extended_love()
    #compare_lanczos_extended_love(dtype=torch.float32)
    #compare_lanczos_extended_love(device="mps", dtype=torch.float32)
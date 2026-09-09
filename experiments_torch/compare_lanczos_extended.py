import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src_torch.corrections import exact_correction, love_correction
from src_torch.diagnostics import projection_residual, relative_error
from src_torch.gp import GP
from src_torch.kernels import RBFKernel
from src_torch.linalg.cg import linear_cg, cg_store_lanczos_basis
from src_torch.linalg.lanczos import lanczos_tridiag, extend_lanczos_basis
from src_torch.means import ZeroMean

import warnings

warnings.filterwarnings("ignore")


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _elapsed(start: float, end: float) -> float:
    return end - start


def compare_lanczos_extended_love(
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
    extension_tol: float = 1e-12,
    view: str = "both",
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> None:
    """Compares CG-based Lanczos extension against LOVE in PyTorch.

    LOVE uses a plain CG solve for the predictive mean and a separate Lanczos
    decomposition for the covariance correction. The extended method stores CG
    search directions, converts them to an orthonormal basis by QR, and extends
    this basis by additional Lanczos steps.

    :param int n: Number of training points. (Default: `1000`.)
    :param int m: Number of test points. (Default: `100`.)
    :param int cg_J: Maximum number of CG iterations. (Default: `100`.)
    :param int lanczos_J: Maximum number of Lanczos iterations for LOVE. (Default: `100`.)
    :param float outputscale: Kernel outputscale. (Default: `1.0`.)
    :param list lengthscales: Lengthscales used in the experiment.
    :param list noises: Noise levels used in the experiment.
    :param list jitters: Jitter values used in approximate covariance corrections.
    :param tuple domain: Interval from which training and test inputs are sampled. (Default: `(-10.0, 10.0)`.)
    :param int seed: Random seed. (Default: `123`.)
    :param float extension_tol: Breakdown tolerance for basis extension. (Default: `1e-12`.)
    :param str view: Output mode. Must be `"accuracy"`, `"time"`, or `"both"`. (Default: `"both"`.)
    :param torch.dtype dtype: Tensor dtype. (Default: `torch.float64`.)
    :param device: Torch device. (Default: `"cpu"`.)
    """
    view = view.lower()
    if view not in {"accuracy", "time", "both"}:
        raise ValueError("view must be one of 'accuracy', 'time', or 'both'.")

    device = torch.device(device)

    if lengthscales is None:
        lengthscales = [0.1, 0.3, 1.0, 3.0, 10.0]

    if noises is None:
        noises = [1e-4, 1e-2, 1.0]

    if jitters is None:
        jitters = [1e-8, 1e-6, 1e-4]

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
                    exact_var = torch.diag(exact_cov)

                    rhs = gp_exact.centered_y.reshape(1, -1, 1)

                    _sync_if_needed(device)
                    t0 = time.perf_counter()
                    _, Q_resid, T_resid = cg_store_lanczos_basis(
                        lambda v: torch.matmul(gp_exact.K_noise, v),
                        rhs,
                        tolerance=1e-5,
                        max_iter=cg_J,
                    )
                    _sync_if_needed(device)
                    t1 = time.perf_counter()

                    _sync_if_needed(device)
                    t2 = time.perf_counter()
                    _ = linear_cg(
                        lambda v: torch.matmul(gp_exact.K_noise, v),
                        rhs,
                        tolerance=1e-5,
                        max_iter=cg_J,
                    )
                    _sync_if_needed(device)
                    t3 = time.perf_counter()
                    _sync_if_needed(device)
                    t4 = time.perf_counter()
                    Q_love, T_love = lanczos_tridiag(
                        lambda v: torch.matmul(gp_exact.K_noise, v),
                        max_iter=lanczos_J,
                        dtype=dtype,
                        device=device,
                        matrix_shape=gp_exact.K_noise.shape,
                        batch_shape=rhs.shape[:-2],
                        init_vecs=rhs,
                        num_init_vecs=1,
                        tol=1e-6,
                    )
                    _sync_if_needed(device)
                    t5 = time.perf_counter()

                    target_J = max(Q_resid.shape[-1], Q_love.shape[-1])

                    _sync_if_needed(device)
                    t6 = time.perf_counter()
                    Q_ext, T_ext = extend_lanczos_basis(
                        lambda v: torch.matmul(gp_exact.K_noise, v),
                        max_iter=target_J,
                        dtype=dtype,
                        device=device,
                        matrix_shape=gp_exact.K_noise.shape,
                        q_mat=Q_resid,
                        t_mat=T_resid,
                        tol=extension_tol,
                    )
                    _sync_if_needed(device)
                    t7 = time.perf_counter()

                    Q_ext_2d = Q_ext.squeeze(0)
                    T_ext_2d = T_ext.squeeze(0)
                    Q_love_2d = Q_love.squeeze(0)
                    T_love_2d = T_love.squeeze(0)

                    proj_ext = projection_residual(Q_ext_2d, k)
                    proj_love = projection_residual(Q_love_2d, k)

                    time_resid_fit = _elapsed(t0, t1)
                    time_love_cg_fit = _elapsed(t2, t3)
                    time_love_fit = _elapsed(t4, t5)
                    time_extend = _elapsed(t6, t7)


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

                            ext_var = torch.diag(ext_cov)
                            love_var = torch.diag(love_cov)

                            time_ext_corr = _elapsed(t8, t9)
                            time_love_corr = _elapsed(t9, t10)

                            total_extend = time_resid_fit + time_extend + time_ext_corr
                            total_love = time_love_cg_fit + time_love_fit + time_love_corr
                            rel_diff_extend_love = (total_extend - total_love) / total_love

                            if view in {"accuracy", "both"}:
                                print(
                                    f"{lengthscale:<10.1g} "
                                    f"{noise:<10.1e} "
                                    f"{jitter:<9.1e} "
                                    f"{Q_resid.shape[-1]:<7d} "
                                    f"{Q_ext.shape[-1]:<5d} "
                                    f"{Q_love.shape[-1]:<7d} "
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
                                    f"{Q_resid.shape[-1]:<7d} "
                                    f"{Q_ext.shape[-1]:<5d} "
                                    f"{Q_love.shape[-1]:<7d} "
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
    compare_lanczos_extended_love(view="accuracy")
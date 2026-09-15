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

from src_torch.gp import GP
from src_torch.kernels import RBFKernel
from src_torch.means import ZeroMean
from src_torch.linalg.cg import cg_store_lanczos_basis
from src_torch.linalg.lanczos import lanczos_tridiag, extend_lanczos_basis
from src_torch.corrections import exact_correction, love_correction

warnings.filterwarnings("ignore")


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def timed(label, device, fn):
    sync_if_needed(device)
    start = time.perf_counter()
    out = fn()
    sync_if_needed(device)
    end = time.perf_counter()
    return out, end - start


def matmul_basis(matmul_closure, q_mat):
    kq_mat = torch.empty_like(q_mat)

    for j in range(q_mat.size(-1)):
        q_j = q_mat[..., :, j : j + 1]
        kq_mat[..., :, j : j + 1].copy_(matmul_closure(q_j))

    return kq_mat


def relative_error(approx, exact):
    return torch.linalg.vector_norm(approx - exact) / torch.linalg.vector_norm(exact)


def run_case(
    lengthscale,
    noise,
    jitter=1e-6,
    n_train=10_000,
    n_test=100,
    max_iter=500,
    device="cpu",
    dtype=torch.float64,
    seed=0,
):
    device = torch.device(device)
    torch.manual_seed(seed)

    X_train = torch.linspace(-3.0, 3.0, n_train, device=device, dtype=dtype).unsqueeze(-1)
    y_train = torch.sin(X_train.squeeze(-1))

    X_test = torch.linspace(-2.5, 2.5, n_test, device=device, dtype=dtype).unsqueeze(-1)

    kernel = RBFKernel(lengthscale=lengthscale, outputscale=1.0)
    mean = ZeroMean()

    gp_exact = GP(X_train, y_train, kernel, mean, noise=noise)
    gp_exact.compute_posterior(method="exact")

    K_noise = gp_exact.K_noise
    K_test = gp_exact.prior_covariance(X_test)
    k = gp_exact.train_test_covariance(X_test)
    rhs = gp_exact._rhs()

    exact_cov = K_test - exact_correction(K_noise, k)

    def matmul_closure(v):
        return torch.matmul(K_noise.unsqueeze(0), v)

    print()
    print(f"case lengthscale={lengthscale}, noise={noise}, jitter={jitter}")
    print("-" * 80)

    # EXT path
    (res_store, Q_resid, KQ_resid), t_cg_store = timed(
        "cg_store",
        device,
        lambda: cg_store_lanczos_basis(
            matmul_closure,
            rhs,
            tolerance=1e-10 if dtype == torch.float64 else 1e-5,
            eps=1e-12 if dtype == torch.float64 else 1e-6,
            stop_updating_after=1e-10 if dtype == torch.float64 else 1e-5,
            max_iter=max_iter,
        ),
    )

    (_, t_resid_build) = timed(
        "T_resid",
        device,
        lambda: torch.matmul(Q_resid.transpose(-1, -2), KQ_resid),
    )

    (Q_ext, T_ext), t_extend = timed(
        "extend",
        device,
        lambda: extend_lanczos_basis(
            matmul_closure,
            max_iter,
            dtype,
            device,
            K_noise.shape,
            Q_resid,
            KQ_resid,
            tol=1e-12 if dtype == torch.float64 else 1e-6,
        ),
    )

    ext_cov, t_ext_corr = timed(
        "ext_correction",
        device,
        lambda: K_test - love_correction(Q_ext.squeeze(0), T_ext.squeeze(0), k, jitter=jitter),
    )

    # LOVE path
    (Q_love, T_love), t_love_basis = timed(
        "love_basis",
        device,
        lambda: lanczos_tridiag(
            matmul_closure,
            max_iter,
            dtype,
            device,
            K_noise.shape,
            batch_shape=torch.Size([1]),
            init_vecs=rhs,
            num_init_vecs=1,
            tol=1e-12 if dtype == torch.float64 else 1e-6,
        ),
    )

    love_cov, t_love_corr = timed(
        "love_correction",
        device,
        lambda: K_test - love_correction(Q_love.squeeze(0), T_love.squeeze(0), k, jitter=jitter),
    )

    total_ext = t_cg_store + t_resid_build + t_extend + t_ext_corr
    total_love = t_love_basis + t_love_corr

    print(f"J_resid: {Q_resid.shape[-1]}")
    print(f"J_ext:   {Q_ext.shape[-1]}")
    print(f"J_love:  {Q_love.shape[-1]}")
    print()
    print("EXT timings")
    print(f"  cg_store:       {t_cg_store:.4e}")
    print(f"  T_resid_build:  {t_resid_build:.4e}")
    print(f"  extend:         {t_extend:.4e}")
    print(f"  correction:     {t_ext_corr:.4e}")
    print(f"  total:          {total_ext:.4e}")
    print()
    print("LOVE timings")
    print(f"  basis:          {t_love_basis:.4e}")
    print(f"  correction:     {t_love_corr:.4e}")
    print(f"  total:          {total_love:.4e}")
    print()
    print("accuracy")
    print(f"  rel_ext_exact:  {relative_error(ext_cov, exact_cov):.4e}")
    print(f"  rel_love_exact: {relative_error(love_cov, exact_cov):.4e}")
    print(f"  rel_diff_time:  {(total_ext - total_love) / total_love:.4e}")


def main():
    cases = [
        (0.3, 1.0),
        (3.0, 1e-2),
        (10.0, 1e-4),
        (1.0, 1.0),
    ]

    for lengthscale, noise in cases:
        run_case(
            lengthscale=lengthscale,
            noise=noise,
            jitter=1e-6,
            n_train=10_000,
            n_test=100,
            max_iter=500,
            device="cpu",
            dtype=torch.float64,
        )


if __name__ == "__main__":
    main()


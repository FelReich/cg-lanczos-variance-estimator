from __future__ import annotations

import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src_torch.linalg.cg import cg_store_lanczos_basis
from src_torch.linalg.lanczos import lanczos_tridiag
from src_torch.corrections import exact_correction, love_correction
from src_torch.kernels import RBFKernel


def relative_error(approx: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
    denom = torch.linalg.matrix_norm(exact)
    if denom == 0:
        return torch.linalg.matrix_norm(approx - exact)
    return torch.linalg.matrix_norm(approx - exact) / denom


def subspace_diagnostics(Q_a: torch.Tensor, Q_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Q_a, Q_b: [batch, n, J]
    M = Q_a.transpose(-1, -2) @ Q_b
    singular_values = torch.linalg.svdvals(M)

    min_cos = singular_values.min(dim=-1).values

    P_a = Q_a @ Q_a.transpose(-1, -2)
    P_b = Q_b @ Q_b.transpose(-1, -2)

    defect = (
        torch.linalg.matrix_norm(P_a - P_b)
        / torch.linalg.matrix_norm(P_b)
    )

    return min_cos, defect


def aligned_t_error(Q_a: torch.Tensor, T_a: torch.Tensor, Q_b: torch.Tensor, T_b: torch.Tensor) -> torch.Tensor:
    # Align basis a to basis b by the overlap matrix.
    # If the subspaces agree, S is nearly orthogonal and S.T @ T_a @ S should match T_b.
    S = Q_a.transpose(-1, -2) @ Q_b
    T_a_aligned = S.transpose(-1, -2) @ T_a @ S

    return torch.linalg.matrix_norm(T_a_aligned - T_b) / torch.linalg.matrix_norm(T_b)


def run_case(
    *,
    n: int = 1000,
    n_test: int = 100,
    lengthscale: float = 0.1,
    noise: float = 1e-4,
    jitter: float = 1e-6,
    max_cg_iter: int = 500,
    max_lanczos_iter: int = 500,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> None:
    device = torch.device(device)
    torch.manual_seed(seed)

    print()
    print("# CASE")
    print(f"n: {n}")
    print(f"lengthscale: {lengthscale}")
    print(f"noise: {noise}")
    print(f"jitter: {jitter}")
    print(f"dtype: {dtype}")
    print(f"device: {device}")
    print()

    X_train = torch.linspace(0.0, 1.0, n, dtype=dtype, device=device).unsqueeze(-1)
    X_test = torch.linspace(0.05, 0.95, n_test, dtype=dtype, device=device).unsqueeze(-1)

    y_train = (
        torch.sin(2.0 * torch.pi * X_train.squeeze(-1))
        + 0.25 * torch.cos(6.0 * torch.pi * X_train.squeeze(-1))
    ).unsqueeze(0).unsqueeze(-1)

    kernel = RBFKernel(lengthscale=lengthscale, outputscale=1.0)

    K = kernel(X_train, X_train)
    K_noise = K + noise * torch.eye(n, dtype=dtype, device=device)
    K_test = kernel(X_train, X_test)

    def matmul_closure(x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(K_noise, x)

    exact_cov = exact_correction(K_noise, K_test)

    # -------------------------------------------------------------------------
    # CG-derived basis and projected matrix.
    # -------------------------------------------------------------------------
    result_cg, Q_cg, T_cg = cg_store_lanczos_basis(
        matmul_closure,
        y_train,
        tolerance=1e-10 if dtype == torch.float64 else 1e-6,
        eps=1e-14 if dtype == torch.float64 else 1e-6,
        stop_updating_after=1e-10 if dtype == torch.float64 else 1e-6,
        max_iter=max_cg_iter,
    )

    J_cg = Q_cg.shape[-1]

    # Direct T diagnostic for CG basis.
    KQ_cg_direct = matmul_closure(Q_cg)
    T_cg_direct = Q_cg.transpose(-1, -2) @ KQ_cg_direct
    T_cg_direct = 0.5 * (T_cg_direct + T_cg_direct.transpose(-1, -2))

    # -------------------------------------------------------------------------
    # LOVE basis, truncated to the same number of iterations.
    # -------------------------------------------------------------------------
    Q_love_full, T_love_full = lanczos_tridiag(
        matmul_closure,
        max(max_lanczos_iter, J_cg),
        dtype,
        device,
        K_noise.shape,
        batch_shape=torch.Size([1]),
        init_vecs=y_train.contiguous(),
        num_init_vecs=1,
        tol=1e-6 if dtype == torch.float32 else 1e-12,
    )

    Q_love = Q_love_full[..., :, :J_cg].contiguous()
    T_love = T_love_full[..., :J_cg, :J_cg].contiguous()

    KQ_love_direct = matmul_closure(Q_love)
    T_love_direct = Q_love.transpose(-1, -2) @ KQ_love_direct
    T_love_direct = 0.5 * (T_love_direct + T_love_direct.transpose(-1, -2))

    # -------------------------------------------------------------------------
    # Corrections at the same iteration count.
    # -------------------------------------------------------------------------
    cg_cov = love_correction(
        Q_cg.squeeze(0),
        T_cg.squeeze(0),
        K_test,
        jitter=jitter,
    )

    cg_direct_t_cov = love_correction(
        Q_cg.squeeze(0),
        T_cg_direct.squeeze(0),
        K_test,
        jitter=jitter,
    )

    love_same_cov = love_correction(
        Q_love.squeeze(0),
        T_love.squeeze(0),
        K_test,
        jitter=jitter,
    )

    # -------------------------------------------------------------------------
    # Diagnostics.
    # -------------------------------------------------------------------------
    eye_cg = torch.eye(J_cg, dtype=dtype, device=device).unsqueeze(0)

    Q_cg_orth = torch.linalg.matrix_norm(Q_cg.transpose(-1, -2) @ Q_cg - eye_cg)
    Q_love_orth = torch.linalg.matrix_norm(Q_love.transpose(-1, -2) @ Q_love - eye_cg)

    T_cg_direct_err = (
        torch.linalg.matrix_norm(T_cg - T_cg_direct)
        / torch.linalg.matrix_norm(T_cg_direct)
    )

    T_love_direct_err = (
        torch.linalg.matrix_norm(T_love - T_love_direct)
        / torch.linalg.matrix_norm(T_love_direct)
    )

    min_cos, subspace_defect = subspace_diagnostics(Q_cg, Q_love)
    T_aligned_err = aligned_t_error(Q_cg, T_cg, Q_love, T_love)

    cg_vs_love_cov = relative_error(cg_cov, love_same_cov)
    cg_direct_t_vs_love_cov = relative_error(cg_direct_t_cov, love_same_cov)

    print("Shapes")
    print("Q_cg:", Q_cg.shape, "T_cg:", T_cg.shape)
    print("Q_love same:", Q_love.shape, "T_love same:", T_love.shape)
    print()

    print("Iterations")
    print("J_cg:", J_cg)
    print()

    print("Basis diagnostics")
    print("Q_cg orth:", Q_cg_orth)
    print("Q_love orth:", Q_love_orth)
    print("min principal cosine:", min_cos)
    print("subspace defect:", subspace_defect)
    print()

    print("T diagnostics")
    print("T_cg/direct:", T_cg_direct_err)
    print("T_love/direct:", T_love_direct_err)
    print("aligned T error:", T_aligned_err)
    print("min eig T_cg:", torch.linalg.eigvalsh(T_cg).min())
    print("min eig T_cg_direct:", torch.linalg.eigvalsh(T_cg_direct).min())
    print("min eig T_love:", torch.linalg.eigvalsh(T_love).min())
    print()

    print("Covariance accuracy")
    print("rel_cg_exact:", relative_error(cg_cov, exact_cov))
    print("rel_cg_direct_t_exact:", relative_error(cg_direct_t_cov, exact_cov))
    print("rel_love_same_exact:", relative_error(love_same_cov, exact_cov))
    print("rel_cg_love_same:", cg_vs_love_cov)
    print("rel_cg_direct_t_love_same:", cg_direct_t_vs_love_cov)


def main() -> None:
    cases = [
        dict(lengthscale=0.1, noise=1e-4),
        dict(lengthscale=0.1, noise=1e-2),
        dict(lengthscale=0.3, noise=1e-4),
        dict(lengthscale=1.0, noise=1e-4),
        dict(lengthscale=3.0, noise=1e-4),
        dict(lengthscale=10.0, noise=1e-4),
    ]

    for case in cases:
        run_case(
            n=1000,
            n_test=100,
            jitter=1e-6,
            max_cg_iter=500,
            max_lanczos_iter=500,
            dtype=torch.float64,
            device="cpu",
            **case,
        )


if __name__ == "__main__":
    main()
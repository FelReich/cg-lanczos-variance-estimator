from __future__ import annotations

import torch

from src_torch.linalg.cg import cg_store_lanczos_basis
from src_torch.corrections import exact_correction, love_correction
from src_torch.linalg.lanczos import lanczos_tridiag, extend_lanczos_basis
from src_torch.diagnostics import relative_error


def make_spd_matrix(n: int, dtype: torch.dtype) -> torch.Tensor:
    torch.manual_seed(123)

    X = torch.linspace(-3.0, 3.0, n, dtype=dtype).reshape(-1, 1)
    sq_dist = (X - X.T) ** 2

    lengthscale = 0.5
    noise = 1e-2

    K = torch.exp(-sq_dist / (2.0 * lengthscale**2))
    K = K + noise * torch.eye(n, dtype=dtype)

    return K


def main() -> None:
    dtype = torch.float64
    device = torch.device("cpu")

    n = 300
    m = 40
    cg_iter = 80
    love_iter = 120
    target_iter = 120
    tol = 1e-10
    jitter = 1e-8

    K = make_spd_matrix(n, dtype=dtype).to(device)

    X_test = torch.linspace(-2.5, 2.5, m, dtype=dtype, device=device).reshape(-1, 1)
    X_train = torch.linspace(-3.0, 3.0, n, dtype=dtype, device=device).reshape(-1, 1)

    sq_dist_test_train = (X_train - X_test.T) ** 2
    k = torch.exp(-sq_dist_test_train / (2.0 * 0.5**2))

    y = torch.sin(X_train.squeeze(-1))
    rhs = y.reshape(1, n, 1)

    def matmul_closure(v: torch.Tensor) -> torch.Tensor:
        if v.dim() == 2:
            return torch.matmul(K, v)
        return torch.matmul(K.unsqueeze(0), v)

    result_cg, q_cg, t_cg = cg_store_lanczos_basis(
        matmul_closure,
        rhs,
        tolerance=tol,
        eps=1e-14,
        stop_updating_after=1e-14,
        max_iter=cg_iter,
    )

    q_love, t_love = lanczos_tridiag(
        matmul_closure,
        max_iter=love_iter,
        dtype=dtype,
        device=device,
        matrix_shape=K.shape,
        batch_shape=torch.Size([1]),
        init_vecs=rhs,
        num_init_vecs=1,
        tol=tol,
    )

    q_ext, t_ext = extend_lanczos_basis(
        matmul_closure,
        max_iter=target_iter,
        dtype=dtype,
        device=device,
        matrix_shape=K.shape,
        q_mat=q_cg,
        t_mat=t_cg,
        tol=tol,
    )

    q_cg_2d = q_cg.squeeze(0)
    q_love_2d = q_love.squeeze(0)
    q_ext_2d = q_ext.squeeze(0)

    t_cg_2d = t_cg.squeeze(0)
    t_love_2d = t_love.squeeze(0)
    t_ext_2d = t_ext.squeeze(0)

    eye_cg = torch.eye(q_cg_2d.shape[-1], dtype=dtype, device=device)
    eye_love = torch.eye(q_love_2d.shape[-1], dtype=dtype, device=device)
    eye_ext = torch.eye(q_ext_2d.shape[-1], dtype=dtype, device=device)

    t_cg_direct = q_cg_2d.T @ K @ q_cg_2d
    t_cg_direct = 0.5 * (t_cg_direct + t_cg_direct.T)

    t_love_direct = q_love_2d.T @ K @ q_love_2d
    t_love_direct = 0.5 * (t_love_direct + t_love_direct.T)

    t_ext_direct = q_ext_2d.T @ K @ q_ext_2d
    t_ext_direct = 0.5 * (t_ext_direct + t_ext_direct.T)

    exact_corr = exact_correction(K, k)
    cg_corr = love_correction(q_cg_2d, t_cg_2d, k, jitter=jitter)
    love_corr = love_correction(q_love_2d, t_love_2d, k, jitter=jitter)
    ext_corr = love_correction(q_ext_2d, t_ext_2d, k, jitter=jitter)

    print("Shapes")
    print("q_cg:", q_cg.shape, "t_cg:", t_cg.shape)
    print("q_love:", q_love.shape, "t_love:", t_love.shape)
    print("q_ext:", q_ext.shape, "t_ext:", t_ext.shape)

    print("\nOrthogonality")
    print("cg:", torch.linalg.matrix_norm(q_cg_2d.T @ q_cg_2d - eye_cg))
    print("love:", torch.linalg.matrix_norm(q_love_2d.T @ q_love_2d - eye_love))
    print("ext:", torch.linalg.matrix_norm(q_ext_2d.T @ q_ext_2d - eye_ext))

    print("\nT consistency")
    print("cg:", torch.linalg.matrix_norm(t_cg_2d - t_cg_direct) / torch.linalg.matrix_norm(t_cg_direct))
    print("love:", torch.linalg.matrix_norm(t_love_2d - t_love_direct) / torch.linalg.matrix_norm(t_love_direct))
    print("ext:", torch.linalg.matrix_norm(t_ext_2d - t_ext_direct) / torch.linalg.matrix_norm(t_ext_direct))

    print("\nMin eigenvalues")
    print("cg:", torch.linalg.eigvalsh(t_cg_2d).min())
    print("love:", torch.linalg.eigvalsh(t_love_2d).min())
    print("ext:", torch.linalg.eigvalsh(t_ext_2d).min())

    print("\nCorrection errors")
    print("cg:", relative_error(cg_corr, exact_corr))
    print("love:", relative_error(love_corr, exact_corr))
    print("ext:", relative_error(ext_corr, exact_corr))

    assert q_cg.shape[0] == 1
    assert q_love.shape[0] == 1
    assert q_ext.shape[0] == 1

    assert torch.linalg.matrix_norm(q_cg_2d.T @ q_cg_2d - eye_cg) < 1e-8
    assert torch.linalg.matrix_norm(q_love_2d.T @ q_love_2d - eye_love) < 1e-8
    assert torch.linalg.matrix_norm(q_ext_2d.T @ q_ext_2d - eye_ext) < 1e-8

    assert torch.linalg.eigvalsh(t_cg_2d).min() > 0
    assert torch.linalg.eigvalsh(t_love_2d).min() > 0
    assert torch.linalg.eigvalsh(t_ext_2d).min() > 0

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
from __future__ import annotations

import time
import warnings

import torch

from src_torch.gp import GP
from src_torch.kernels import RBFKernel
from src_torch.means import ZeroMean
from src_torch.linalg.cg import cg_store_lanczos_basis
from src_torch.linalg.lanczos import lanczos_tridiag, extend_lanczos_basis
from src_torch.corrections import exact_correction, love_correction

warnings.filterwarnings("ignore")



import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src_torch.linalg.cg import linear_cg
from src_torch.linalg.lanczos import lanczos_tridiag


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def timed(device: torch.device, fn):
    sync_if_needed(device)
    start = time.perf_counter()
    out = fn()
    sync_if_needed(device)
    return out, time.perf_counter() - start


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
    (res_store, Q_resid), t_cg_store = timed(
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

    (Q_ext, T_ext), t_extend = timed(
        device,
        lambda: extend_lanczos_basis(
            matmul_closure,
            max_iter,
            dtype,
            device,
            K_noise.shape,
            Q_resid,
            tol=1e-12 if dtype == torch.float64 else 1e-6,
        ),
    )

    ext_cov, t_ext_corr = timed(
        device,
        lambda: K_test - love_correction(
            Q_ext.squeeze(0),
            T_ext.squeeze(0),
            k,
            jitter=jitter,
        ),
    )

    # LOVE path
    (Q_love, T_love), t_love_basis = timed(
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
        device,
        lambda: K_test - love_correction(
            Q_love.squeeze(0),
            T_love.squeeze(0),
            k,
            jitter=jitter,
        ),
    )

    total_ext = t_cg_store + t_extend + t_ext_corr
    total_love = t_love_basis + t_love_corr

    print(f"J_resid: {Q_resid.shape[-1]}")
    print(f"J_ext:   {Q_ext.shape[-1]}")
    print(f"J_love:  {Q_love.shape[-1]}")
    print()
    print("EXT timings")
    print(f"  cg_store:    {t_cg_store:.4e}")
    print(f"  extend:      {t_extend:.4e}")
    print(f"  correction:  {t_ext_corr:.4e}")
    print(f"  total:       {total_ext:.4e}")
    print()
    print("LOVE timings")
    print(f"  basis:       {t_love_basis:.4e}")
    print(f"  correction:  {t_love_corr:.4e}")
    print(f"  total:       {total_love:.4e}")
    print()
    print("accuracy")
    print(f"  rel_ext_exact:   {relative_error(ext_cov, exact_cov):.4e}")
    print(f"  rel_love_exact:  {relative_error(love_cov, exact_cov):.4e}")
    print(f"  rel_diff_time:   {(total_ext - total_love) / total_love:.4e}")


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






def rbf_kernel(X1, X2, lengthscale=1.0, outputscale=1.0):
    X1 = X1.reshape(-1, 1)
    X2 = X2.reshape(-1, 1)

    sq_dist = (X1[:, None, :] - X2[None, :, :]).pow(2).sum(dim=-1)

    return outputscale**2 * torch.exp(-sq_dist / (2.0 * lengthscale**2))


def matmul_basis(matmul_closure, Q):
    cols = []

    for j in range(Q.size(-1)):
        cols.append(matmul_closure(Q[..., :, j : j + 1]))

    return torch.cat(cols, dim=-1)


def direct_projected_matrix(matmul_closure, Q):
    KQ = matmul_basis(matmul_closure, Q)
    T = Q.transpose(-1, -2).matmul(KQ)

    return 0.5 * (T + T.transpose(-1, -2))


def recover_cg_basis_and_t(matmul_closure, rhs, max_iter, tolerance, eps):
    out = linear_cg(
        matmul_closure,
        rhs,
        tolerance=tolerance,
        eps=eps,
        stop_updating_after=eps,
        max_iter=max_iter,
        save_directions=True,
    )

    if not isinstance(out, tuple):
        raise RuntimeError("linear_cg did not return stored directions.")

    if len(out) == 3:
        result, d_mat, kd_mat = out

        q_mat, r_mat = torch.linalg.qr(d_mat, mode="reduced")

        # This is KQ^T.
        kq_mat_t = torch.linalg.solve(
            r_mat.transpose(-1, -2),
            kd_mat.transpose(-1, -2),
        )

        t_mat = kq_mat_t.matmul(q_mat)
        t_mat = 0.5 * (t_mat + t_mat.transpose(-1, -2))

        kq_direct = matmul_basis(matmul_closure, q_mat)
        t_direct = direct_projected_matrix(matmul_closure, q_mat)

        print("\nCG recovery diagnostics")
        print("-" * 72)
        print("d_mat:", d_mat.shape)
        print("q_mat:", q_mat.shape)
        print("t_mat:", t_mat.shape)
        print("KQ solve/direct error:",
              torch.linalg.matrix_norm(kq_mat_t.transpose(-1, -2) - kq_direct) / torch.linalg.matrix_norm(kq_direct))
        print("T solve/direct error:",
              torch.linalg.matrix_norm(t_mat - t_direct) / torch.linalg.matrix_norm(t_direct))
        print("min eig T solve:", torch.linalg.eigvalsh(t_mat).min())
        print("min eig T direct:", torch.linalg.eigvalsh(t_direct).min())
        print("cond R:", torch.linalg.cond(r_mat))

        return result, q_mat, t_mat

    if len(out) == 2:
        result, d_mat = out

        q_mat, _ = torch.linalg.qr(d_mat, mode="reduced")
        t_mat = direct_projected_matrix(matmul_closure, q_mat)

        print("\nCG recovery diagnostics")
        print("-" * 72)
        print("linear_cg returned only d_mat, so T is computed directly.")
        print("d_mat:", d_mat.shape)
        print("q_mat:", q_mat.shape)
        print("t_mat:", t_mat.shape)
        print("min eig T direct:", torch.linalg.eigvalsh(t_mat).min())

        return result, q_mat, t_mat

    raise RuntimeError(f"Unexpected linear_cg return length: {len(out)}")


def compare_to_lanczos(matmul_closure, Q_cg, T_cg, max_iter, dtype, device, matrix_shape):
    Q_love, T_love = lanczos_tridiag(
        matmul_closure,
        max_iter,
        dtype=dtype,
        device=device,
        matrix_shape=matrix_shape,
        batch_shape=torch.Size([1]),
        init_vecs=Q_cg[..., :, :1],
        num_init_vecs=1,
        tol=1e-6,
    )

    J = min(Q_cg.shape[-1], Q_love.shape[-1])

    Qc = Q_cg[..., :, :J]
    Ql = Q_love[..., :, :J]
    Tc = T_cg[..., :J, :J]
    Tl = T_love[..., :J, :J]

    overlap = Ql.transpose(-1, -2).matmul(Qc)
    svals = torch.linalg.svdvals(overlap)

    C = overlap
    Tl_in_cg_basis = C.transpose(-1, -2).matmul(Tl).matmul(C)

    print("\nCG basis vs Lanczos basis")
    print("-" * 72)
    print("J compare:", J)
    print("min principal cosine:", svals.min())
    print("subspace defect:",
          torch.linalg.matrix_norm(Qc.matmul(Qc.transpose(-1, -2)) - Ql.matmul(Ql.transpose(-1, -2))))
    print("relative aligned T error:",
          torch.linalg.matrix_norm(Tc - Tl_in_cg_basis) / torch.linalg.matrix_norm(Tl_in_cg_basis))
    print("min eig T cg:", torch.linalg.eigvalsh(Tc).min())
    print("min eig T love:", torch.linalg.eigvalsh(Tl).min())


def debug_extend_with_stored_t(matmul_closure, Q0, T0, target_iter, tol=1e-6):
    if Q0.dim() != 3:
        raise ValueError("Q0 must have shape [batch, n, J].")

    batch_shape = Q0.shape[:-2]
    n = Q0.size(-2)
    current_iter = Q0.size(-1)
    dtype = Q0.dtype
    device = Q0.device

    q_ext = Q0.new_zeros(target_iter, *batch_shape, n)
    q_ext[:current_iter].copy_(Q0.permute(-1, *range(len(batch_shape)), -2))

    t_ext = T0.new_zeros(target_iter, target_iter, *batch_shape)
    t_ext[:current_iter, :current_iter].copy_(
        T0.permute(-2, -1, *range(len(batch_shape)))
    )

    print("\nInitial extension state")
    print("-" * 72)
    check_extension_state(matmul_closure, q_ext, t_ext, current_iter, batch_shape, label="initial")

    # Start exactly like the NumPy recurrence version.
    q = q_ext[current_iter - 1]  # [batch, n]
    v = matmul_closure(q.unsqueeze(-1)).squeeze(-1)

    alpha_last = t_ext[current_iter - 1, current_iter - 1]  # [batch]
    v = v - alpha_last.unsqueeze(-1) * q

    if current_iter > 1:
        beta_prev = t_ext[current_iter - 2, current_iter - 1]  # [batch]
        q_prev = q_ext[current_iter - 2]
        v = v - beta_prev.unsqueeze(-1) * q_prev

    # Same two-pass Euclidean reorthogonalization as NumPy.
    for _ in range(2):
        coeffs = (q_ext[:current_iter] * v.unsqueeze(0)).sum(dim=-1)  # [J, batch]
        v = v - (q_ext[:current_iter] * coeffs.unsqueeze(-1)).sum(dim=0)

    num_iter = current_iter

    for j in range(current_iter, target_iter):
        beta = torch.linalg.vector_norm(v, ord=2, dim=-1)  # [batch]

        if torch.sum(beta.abs() > tol) == 0:
            print("breakdown at j =", j)
            break

        t_ext[j - 1, j].copy_(beta)
        t_ext[j, j - 1].copy_(beta)

        q_prev = q
        q = v / beta.clamp_min(torch.finfo(dtype).eps).unsqueeze(-1)

        q_ext[j].copy_(q)
        num_iter = j + 1

        Kq = matmul_closure(q.unsqueeze(-1)).squeeze(-1)

        alpha = (q * Kq).sum(dim=-1)
        t_ext[j, j].copy_(alpha)

        v = Kq - alpha.unsqueeze(-1) * q - beta.unsqueeze(-1) * q_prev

        for _ in range(2):
            coeffs = (q_ext[:num_iter] * v.unsqueeze(0)).sum(dim=-1)
            v = v - (q_ext[:num_iter] * coeffs.unsqueeze(-1)).sum(dim=0)

        if j < current_iter + 10 or j % 25 == 0 or j == target_iter - 1:
            check_extension_state(
                matmul_closure,
                q_ext,
                t_ext,
                num_iter,
                batch_shape,
                label=f"after j={j}",
            )

    q_final = q_ext[:num_iter].permute(*range(1, 1 + len(batch_shape)), -1, 0).contiguous()

    t_raw = t_ext[:num_iter, :num_iter]
    t_final = t_raw.permute(*range(2, 2 + len(batch_shape)), 0, 1).contiguous()
    t_final = 0.5 * (t_final + t_final.transpose(-1, -2))

    t_direct = direct_projected_matrix(matmul_closure, q_final)

    print("\nFinal extension state")
    print("-" * 72)
    print("Q final:", q_final.shape)
    print("T stored:", t_final.shape)
    print("T direct:", t_direct.shape)
    print("final T stored/direct error:",
          torch.linalg.matrix_norm(t_final - t_direct) / torch.linalg.matrix_norm(t_direct))
    print("final min eig stored:", torch.linalg.eigvalsh(t_final).min())
    print("final min eig direct:", torch.linalg.eigvalsh(t_direct).min())

    return q_final, t_final, t_direct


def check_extension_state(matmul_closure, q_ext, t_ext, num_iter, batch_shape, label):
    q_now = q_ext[:num_iter].permute(*range(1, 1 + len(batch_shape)), -1, 0).contiguous()

    t_raw = t_ext[:num_iter, :num_iter]
    t_now = t_raw.permute(*range(2, 2 + len(batch_shape)), 0, 1).contiguous()
    t_now = 0.5 * (t_now + t_now.transpose(-1, -2))

    t_direct = direct_projected_matrix(matmul_closure, q_now)

    eye = torch.eye(num_iter, dtype=q_now.dtype, device=q_now.device).expand(*batch_shape, num_iter, num_iter)

    q_orth = torch.linalg.matrix_norm(q_now.transpose(-1, -2).matmul(q_now) - eye)
    t_err = torch.linalg.matrix_norm(t_now - t_direct) / torch.linalg.matrix_norm(t_direct)

    print(
        label,
        "| J", num_iter,
        "| Q orth", q_orth.item(),
        "| T err", t_err.item(),
        "| min eig stored", torch.linalg.eigvalsh(t_now).min().item(),
        "| min eig direct", torch.linalg.eigvalsh(t_direct).min().item(),
    )


def run_case(
    *,
    n=1000,
    lengthscale=0.1,
    noise=1e-2,
    max_cg_iter=500,
    target_iter=356,
    dtype=torch.float64,
    device="cpu",
):
    device = torch.device(device)

    torch.manual_seed(0)

    X = torch.linspace(-3.0, 3.0, n, dtype=dtype, device=device)
    y = torch.sin(X).reshape(1, n, 1)

    K = rbf_kernel(X, X, lengthscale=lengthscale).to(dtype=dtype, device=device)
    K = K + noise * torch.eye(n, dtype=dtype, device=device)

    K_batch = K.unsqueeze(0)

    def matmul_closure(v):
        return K_batch.matmul(v)

    print("\nCASE")
    print("=" * 72)
    print("n:", n)
    print("lengthscale:", lengthscale)
    print("noise:", noise)
    print("dtype:", dtype)
    print("device:", device)

    result, Q_cg, T_cg = recover_cg_basis_and_t(
        matmul_closure,
        y,
        max_iter=max_cg_iter,
        tolerance=1e-6,
        eps=1e-12 if dtype == torch.float64 else 1e-6,
    )

    compare_to_lanczos(
        matmul_closure,
        Q_cg,
        T_cg,
        max_iter=min(target_iter, n),
        dtype=dtype,
        device=device,
        matrix_shape=torch.Size([n, n]),
    )

    debug_extend_with_stored_t(
        matmul_closure,
        Q_cg,
        T_cg,
        target_iter=min(target_iter, n),
        tol=1e-6,
    )


if __name__ == "__main__":
    run_case(
        n=1000,
        lengthscale=0.1,
        noise=1e-2,
        max_cg_iter=500,
        target_iter=356,
        dtype=torch.float64,
        device="cpu",
    )

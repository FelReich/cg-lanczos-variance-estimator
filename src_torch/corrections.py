from __future__ import annotations

import torch


def exact_correction(K_noise: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    # Computes the exact covariance reduction term k.T @ K_noise^{-1} @ k.
    return k.T @ torch.linalg.solve(K_noise, k)


def love_correction_theory(
    Q: torch.Tensor,
    T: torch.Tensor,
    k: torch.Tensor,
    jitter: float = 1e-6,
    max_tries: int = 5,
) -> torch.Tensor:
    """Compute the LOVE-style covariance reduction term from the theoretical form.

    This version evaluates

        k.T @ Q @ T^{-1} @ Q.T @ k

    using a Cholesky factorization of the projected matrix ``T``. It therefore
    follows the formula used in the theoretical derivation directly. If the
    Cholesky factorization fails, the diagonal jitter is increased geometrically.
    """
    eye = torch.eye(T.size(-1), dtype=T.dtype, device=T.device)

    for i in range(max_tries):
        jitter_i = jitter * (10**i)
        T_jittered = T + jitter_i * eye

        L, info = torch.linalg.cholesky_ex(T_jittered)
        
        if not torch.any(info):
            Z = torch.linalg.solve_triangular(L, Q.T @ k, upper=False)

            return Z.T @ Z

    raise RuntimeError(
        f"Cholesky failed after {max_tries} tries. "
        f"Last jitter was {jitter * (10.0 ** (max_tries - 1)):.1e}."
    )


def love_correction(
    Q: torch.Tensor,
    T: torch.Tensor,
    k: torch.Tensor,
    jitter: float = 1e-6,
) -> torch.Tensor:
    """Compute the LOVE-style covariance reduction as in linear_operator.

    This version mirrors the implementation used by ``linear_operator``:
    a small diagonal jitter proportional to the minimum diagonal entry of ``T``
    is added, then an eigendecomposition of the projected matrix is used to
    construct an inverse-root factor. Non-positive eigenvalues that remain after
    adding jitter are masked out, matching the numerical safeguard used in the
    original implementation.

    The returned matrix is the covariance reduction term

        k.T @ Q @ T^{-1} @ Q.T @ k,

    evaluated through the inverse-root factor rather than through a Cholesky
    solve.
    """
    diag = torch.diagonal(T, dim1=-2, dim2=-1)
    mins = diag.min(dim=-1, keepdim=True).values.unsqueeze(-1)

    eye = torch.eye(T.size(-1), dtype=T.dtype, device=T.device)
    jitter_mat = jitter * mins * eye.expand_as(T)

    evals, evecs = torch.linalg.eigh(T + jitter_mat)

    mask = evals.ge(0)
    evecs = evecs * mask.type_as(evecs).unsqueeze(-2)
    evals = evals.masked_fill_(~mask, 1)

    inv_root = Q.matmul(evecs) / evals.sqrt().unsqueeze(-2)

    Z = inv_root.transpose(-1, -2) @ k
    return Z.transpose(-1, -2) @ Z


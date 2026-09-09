from __future__ import annotations

import torch


def relative_error(approx: torch.Tensor, exact: torch.Tensor) -> float:
    # Computes the relative Frobenius error between an approximation and a reference.
    approx = torch.as_tensor(approx)
    exact = torch.as_tensor(exact, dtype=approx.dtype, device=approx.device)

    denom = torch.linalg.norm(exact)
    if denom == 0:
        return float(torch.linalg.norm(approx - exact).item())

    return float((torch.linalg.norm(approx - exact) / denom).item())


def conjugacy_error(D: torch.Tensor, KD: torch.Tensor) -> float:
    # Measures how far the stored directions are from being K-conjugate.
    D = torch.as_tensor(D)
    KD = torch.as_tensor(KD, dtype=D.dtype, device=D.device)

    if D.shape != KD.shape:
        raise ValueError("D and KD must have the same shape.")

    G = D.T @ KD
    diag_G = torch.diag(torch.diag(G))

    denom = torch.linalg.norm(diag_G)
    if denom == 0:
        return float(torch.linalg.norm(G - diag_G).item())

    return float((torch.linalg.norm(G - diag_G) / denom).item())


def projection_residual(D: torch.Tensor, k: torch.Tensor) -> float:
    # Measures how much of k lies outside the span of the stored directions.
    D = torch.as_tensor(D)
    k = torch.as_tensor(k, dtype=D.dtype, device=D.device)

    if D.ndim != 2:
        raise ValueError("D must be a two-dimensional tensor.")

    if k.ndim == 1:
        k = k.reshape(-1, 1)

    if k.ndim != 2:
        raise ValueError("k must be one- or two-dimensional.")

    if D.shape[0] != k.shape[0]:
        raise ValueError("D and k must have the same first dimension.")

    Q, _ = torch.linalg.qr(D, mode="reduced")
    residual = k - Q @ (Q.T @ k)

    denom = torch.linalg.norm(k)
    if denom == 0:
        return float(torch.linalg.norm(residual).item())

    return float((torch.linalg.norm(residual) / denom).item())
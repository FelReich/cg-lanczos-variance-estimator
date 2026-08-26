from __future__ import annotations

import numpy as np


class ReorthogonalizationRule:
    """Decides when CG search directions should be reorthogonalized.

    The rule is called once per CG iteration and returns True if the current
    direction should be reorthogonalized against previously stored directions.

    :param str mode: Reorthogonalization strategy. Must be one of `"never"`,
        `"always"`, `"every"`, `"rayleigh"`, or `"norm"`. (Default: `"never"`.)
    :param int every: Reorthogonalize every `every` iterations when `mode="every"`.
        (Default: `1`.)
    :param int start: First iteration at which reorthogonalization is allowed.
        (Default: `1`.)
    :param float rayleigh_tol: Threshold for the Rayleigh quotient
        `d.T @ A @ d / d.T @ d` when `mode="rayleigh"`. (Default: `1e-12`.)
    :param float norm_tol: Threshold for the relative direction norm
        `norm(d) / initial_direction_norm` when `mode="norm"`. (Default: `1e-12`.)
    """
    def __init__(
        self,
        mode: str = "never",
        every: int = 1,
        start: int = 1,
        rayleigh_tol: float = 1e-12,
        norm_tol: float = 1e-12,
    ):
        self.mode = mode.lower()
        self.every = int(every)
        self.start = int(start)
        self.rayleigh_tol = float(rayleigh_tol)
        self.norm_tol = float(norm_tol)

        if self.mode not in {"never", "always", "every", "rayleigh", "norm"}:
            raise ValueError("mode must be one of 'never', 'always', 'every', 'rayleigh', or 'norm'.")

        if self.every <= 0:
            raise ValueError("every must be positive.")

        if self.rayleigh_tol <= 0:
            raise ValueError("rayleigh_tol must be positive.")

        if self.norm_tol <= 0:
            raise ValueError("norm_tol must be positive.")
        
        if self.start < 0:
            raise ValueError("start must be nonnegative.")

    def __call__(
        self,
        *,
        i: int,
        d: np.ndarray,
        v: np.ndarray,
        dot_dv: float,
        initial_direction_norm: float,
    ) -> bool:
        #Returns whether the current CG direction should be reorthogonalized
        if i < self.start:
            return False
        
        match self.mode:
            case "never":
                return False

            case "always":
                return True

            case "every":
                return i % self.every == 0

            case "rayleigh":
                d_norm_sq = np.dot(d, d)

                if d_norm_sq == 0:
                    return True

                return dot_dv / d_norm_sq < self.rayleigh_tol

            case "norm":
                d_norm = np.linalg.norm(d)

                if initial_direction_norm == 0:
                    return True

                return d_norm / initial_direction_norm < self.norm_tol

            case _:
                raise RuntimeError("Invalid reorthogonalization rule state.")
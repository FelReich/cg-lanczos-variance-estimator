from __future__ import annotations

import numpy as np


class ReorthogonalizationRule:
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

    def __call__(
        self,
        *,
        i: int,
        d: np.ndarray,
        v: np.ndarray,
        dot_dv: float,
        initial_direction_norm: float,
    ) -> bool:
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
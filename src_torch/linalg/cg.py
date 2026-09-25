#!/usr/bin/env python3
from __future__ import annotations

import warnings

import torch


def _default_preconditioner(x):
    return x.clone()


@torch.jit.script
def _jit_linear_cg_updates(
    result,
    alpha,
    residual_inner_prod,
    eps,
    beta,
    residual,
    precond_residual,
    mul_storage,
    is_zero,
    curr_conjugate_vec,
):
    # # Update result
    # # result_{k} = result_{k-1} + alpha_{k} p_vec_{k-1}
    result = torch.addcmul(result, alpha, curr_conjugate_vec, out=result)

    # beta_{k} = (precon_residual{k}^T r_vec_{k}) / (precon_residual{k-1}^T r_vec_{k-1})
    beta.resize_as_(residual_inner_prod).copy_(residual_inner_prod)
    torch.mul(residual, precond_residual, out=mul_storage)
    torch.sum(mul_storage, -2, keepdim=True, out=residual_inner_prod)

    # Do a safe division here
    torch.lt(beta, eps, out=is_zero)
    beta.masked_fill_(is_zero, 1)
    torch.div(residual_inner_prod, beta, out=beta)
    beta.masked_fill_(is_zero, 0)

    # Update curr_conjugate_vec
    # curr_conjugate_vec_{k} = precon_residual{k} + beta_{k} curr_conjugate_vec_{k-1}
    curr_conjugate_vec.mul_(beta).add_(precond_residual)


@torch.jit.script
def _jit_linear_cg_updates_no_precond(
    mvms,
    result,
    has_converged,
    alpha,
    residual_inner_prod,
    eps,
    beta,
    residual,
    precond_residual,
    mul_storage,
    is_zero,
    curr_conjugate_vec,
):
    torch.mul(curr_conjugate_vec, mvms, out=mul_storage)
    torch.sum(mul_storage, dim=-2, keepdim=True, out=alpha)

    # Do a safe division here
    torch.lt(alpha, eps, out=is_zero)
    alpha.masked_fill_(is_zero, 1)
    torch.div(residual_inner_prod, alpha, out=alpha)
    alpha.masked_fill_(is_zero, 0)

    # We'll cancel out any updates by setting alpha=0 for any vector that has already converged
    alpha.masked_fill_(has_converged, 0)

    # Update residual
    # residual_{k} = residual_{k-1} - alpha_{k} mat p_vec_{k-1}
    torch.addcmul(residual, -alpha, mvms, out=residual)

    # Update precond_residual
    # precon_residual{k} = M^-1 residual_{k}
    precond_residual = residual.clone()

    _jit_linear_cg_updates(
        result,
        alpha,
        residual_inner_prod,
        eps,
        beta,
        residual,
        precond_residual,
        mul_storage,
        is_zero,
        curr_conjugate_vec,
    )


@torch.jit.script
def _jit_linear_cg_updates_save_directions(
    mvms,
    result,
    has_converged,
    alpha,
    residual_inner_prod,
    eps,
    beta,
    residual,
    precond_residual,
    mul_storage,
    is_zero,
    curr_conjugate_vec,
):
    torch.mul(curr_conjugate_vec, mvms, out=mul_storage)
    torch.sum(mul_storage, dim=-2, keepdim=True, out=alpha)

    # Do a safe division here
    torch.lt(alpha, eps, out=is_zero)
    alpha.masked_fill_(is_zero, 1)
    torch.div(torch.mul(residual, curr_conjugate_vec).sum(dim=-2, keepdim=True), alpha, out=alpha)
    alpha.masked_fill_(is_zero, 0)

    # We'll cancel out any updates by setting alpha=0 for any vector that has already converged
    alpha.masked_fill_(has_converged, 0)

    # Update residual
    # residual_{k} = residual_{k-1} - alpha_{k} mat p_vec_{k-1}
    torch.addcmul(residual, -alpha, mvms, out=residual)

    # Update precond_residual
    # precon_residual{k} = M^-1 residual_{k}
    precond_residual = residual.clone()

    _jit_linear_cg_updates(
        result,
        alpha,
        residual_inner_prod,
        eps,
        beta,
        residual,
        precond_residual,
        mul_storage,
        is_zero,
        curr_conjugate_vec,
    )


def linear_cg(
    matmul_closure,
    rhs,
    n_tridiag=0, # By default, n_tridiag = 0.
    tolerance=1e-10,
    eps=1e-10,
    stop_updating_after=1e-10,
    max_iter=100,
    max_tridiag_iter=0, # By default, max_tridiag_iter = 0.
    initial_guess=None,
    preconditioner=None, # By default, no preconditioner is used.
    save_directions=False,
):
    """Run conjugate gradients, optionally storing reorthogonalized CG directions.

    This is a prototype version of the ``linear_operator`` conjugate gradients
    routine. It approximately solves

        A x = b,

    where multiplication by ``A`` is represented by ``matmul_closure`` and
    ``rhs`` represents ``b``. The system matrix is assumed to be symmetric
    positive definite. The right-hand side is normalized internally before the
    iteration starts and the final result is rescaled before returning.

    Compared with the original routine, this prototype adds the optional
    ``save_directions`` mode. In this mode, the current CG search direction
    ``d_k`` and the corresponding matrix-vector product ``A d_k`` are stored.
    Before a direction is stored, it is reorthogonalized against the previously
    stored directions in the ``A``-inner product. If this reorthogonalization
    fails, direction storage is stopped and the stored prefix is returned.

    In the unpreconditioned stored-direction case, the step size is computed
    using ``r_k.T @ d_k`` rather than the usual stored residual inner product.
    This keeps the residual update consistent after the search direction has
    been modified by reorthogonalization.

    Args:
        matmul_closure: Tensor or callable implementing multiplication by the
            system matrix ``A``. If a tensor is supplied, its ``matmul`` method
            is used.
        rhs: Right-hand side tensor. One-dimensional inputs are treated as a
            single right-hand side and are temporarily unsqueezed.
        n_tridiag: Number of right-hand sides for which a tridiagonalization
            should be computed. 
        tolerance: Relative residual tolerance used for convergence checks and
            for the stored-direction reorthogonalization criterion.
        eps: Small positive value used to avoid division by zero.
        stop_updating_after: Residual norm below which an individual right-hand
            side is treated as converged and no longer updated.
        max_iter: Maximum number of CG iterations.
        max_tridiag_iter: Maximum tridiagonalization size. This is relevant only
            when ``n_tridiag > 0``.
        initial_guess: Optional initial guess for the solution. If omitted, a
            zero tensor with the same shape as ``rhs`` is used.
        preconditioner: Optional callable applying a left preconditioner. If
            omitted, the identity preconditioner is used.
        save_directions: If ``True``, store CG search directions and matrix-
            vector products for later Krylov-basis recovery.

    Returns:
        If ``save_directions`` is ``False`` and ``n_tridiag == 0``, returns the
        approximate solution tensor.

        If ``n_tridiag > 0``, returns the approximate solution together with the
        computed tridiagonal matrices. In this prototype, tridiagonalization 
        through ``n_tridiag`` is currently disabled.

        If ``save_directions`` is ``True``, returns the approximate solution,
        the stored direction matrix, and the stored matrix-vector product matrix.
        The stored matrices have shape ``[*batch_shape, n, J]``, where ``J`` is
        the number of successfully stored directions.
    """

    if n_tridiag:
        raise NotImplementedError("n_tridiag is not supported when storing directions.")
    
    
    # Unsqueeze, if necesasry
    is_vector = rhs.ndimension() == 1
    if is_vector:
        rhs = rhs.unsqueeze(-1)

    # Some default arguments
    if initial_guess is None:
        initial_guess = torch.zeros_like(rhs)
    else:
        # Unsqueeze, if necesasry
        is_vector = initial_guess.ndimension() == 1
        if is_vector:
            initial_guess = initial_guess.unsqueeze(-1)
    if preconditioner is None:
        preconditioner = _default_preconditioner
        precond = False
    else: #No preconditioner will be used, so this part of the code won't ever be reached
        precond = True

    # If we are running m CG iterations, we obviously can't get more than m Lanczos coefficients
    if max_tridiag_iter > max_iter:
        raise RuntimeError("Getting a tridiagonalization larger than the number of CG iterations run is not possible!")

    # Check matmul_closure object
    if torch.is_tensor(matmul_closure):
        matmul_closure = matmul_closure.matmul
    elif not callable(matmul_closure):
        raise RuntimeError("matmul_closure must be a tensor, or a callable object!")

    # Get some constants
    num_rows = rhs.size(-2)
    n_iter = min(max_iter, num_rows) 
    n_tridiag_iter = min(max_tridiag_iter, num_rows)
    eps = torch.tensor(eps, dtype=rhs.dtype, device=rhs.device)

    # Get the norm of the rhs - used for convergence checks
    # Here we're going to make almost-zero norms actually be 1 (so we don't get divide-by-zero issues)
    # But we'll store which norms were actually close to zero
    rhs_norm = rhs.norm(2, dim=-2, keepdim=True)
    rhs_is_zero = rhs_norm.lt(eps)
    rhs_norm = rhs_norm.masked_fill_(rhs_is_zero, 1)

    # Let's normalize. We'll un-normalize afterwards
    rhs = rhs.div(rhs_norm)
    initial_guess = initial_guess.div(rhs_norm)

    # residual: residual_{0} = b_vec - lhs x_{0}
    residual = rhs - matmul_closure(initial_guess)
    batch_shape = residual.shape[:-2]

    # result <- x_{0}
    result = initial_guess.expand_as(residual).contiguous()

    # Check for NaNs
    if not torch.equal(residual, residual):
        raise RuntimeError("NaNs encountered when trying to perform matrix-vector multiplication")

    # Sometime we're lucky and the preconditioner solves the system right away
    # Check for convergence
    residual_norm = residual.norm(2, dim=-2, keepdim=True)
    has_converged = torch.lt(residual_norm, stop_updating_after)

    if has_converged.all() and not n_tridiag:
        n_iter = 0  # Skip the iteration!

    # Otherwise, let's define precond_residual and curr_conjugate_vec
    else:
        # precon_residual{0} = M^-1 residual_{0}
        precond_residual = preconditioner(residual)
        curr_conjugate_vec = precond_residual
        residual_inner_prod = precond_residual.mul(residual).sum(-2, keepdim=True)

        # Define storage matrices
        mul_storage = torch.empty_like(residual)
        alpha = torch.empty(*batch_shape, 1, rhs.size(-1), dtype=residual.dtype, device=residual.device)
        beta = torch.empty_like(alpha)
        is_zero = torch.empty(*batch_shape, 1, rhs.size(-1), dtype=torch.bool, device=residual.device)

    # Define tridiagonal matrices, if applicable
    if n_tridiag:
        t_mat = torch.zeros(
            n_tridiag_iter,
            n_tridiag_iter,
            *batch_shape,
            n_tridiag,
            dtype=alpha.dtype,
            device=alpha.device,
        )
        alpha_tridiag_is_zero = torch.empty(*batch_shape, n_tridiag, dtype=torch.bool, device=t_mat.device)
        alpha_reciprocal = torch.empty(*batch_shape, n_tridiag, dtype=t_mat.dtype, device=t_mat.device)
        prev_alpha_reciprocal = torch.empty_like(alpha_reciprocal)
        prev_beta = torch.empty_like(alpha_reciprocal)

    update_tridiag = True
    last_tridiag_iter = 0

    num_stored = 0
    could_reorthogonalize = True

    # It's conceivable we reach the tolerance on the last iteration, so can't just check iteration number.
    tolerance_reached = False

    # If requested, allocate storage for CG directions and matrix-vector products
    if save_directions:
        d_mat = rhs.new_zeros(n_iter, *batch_shape, num_rows)
        kd_mat = rhs.new_zeros(n_iter, *batch_shape, num_rows)
        save_directions_cg = True
    else:
        save_directions_cg = False

    # Start the iteration
    for k in range(n_iter):
        # Get next alpha
        # alpha_{k} = (residual_{k-1}^T precon_residual{k-1}) / (p_vec_{k-1}^T mat p_vec_{k-1})
        mvms = matmul_closure(curr_conjugate_vec)

        # Store CG directions, if requested
        if save_directions_cg:
            # Reorthogonalize the current direction against previously stored directions
            if k > 0:
                d_prev = d_mat[:k]    
                kd_prev = kd_mat[:k]   

                could_reorthogonalize = False

                for _ in range(10):
                    # Compute d_i^T A d_i for all stored directions
                    dkd = torch.mul(d_prev, kd_prev).sum(dim=-1)
                    dkd_is_zero = torch.lt(dkd.abs(), eps)
                    dkd.masked_fill_(dkd_is_zero, 1.0)

                    # Projection coefficients: (d_i^T A d_k) / (d_i^T A d_i)
                    coeffs = torch.mul(d_prev, mvms.squeeze(-1)).sum(dim=-1).div(dkd)
                    coeffs.masked_fill_(dkd_is_zero, 0.0)  

                    # Remove the projections from d_k and A d_k
                    curr_conjugate_vec.sub_((d_prev * coeffs.unsqueeze(-1)).sum(dim=0).unsqueeze(-1))
                    mvms.sub_((kd_prev * coeffs.unsqueeze(-1)).sum(dim=0).unsqueeze(-1))

                    # Check remaining A-conjugacy relative to the A-norms
                    inner_products = torch.mul(d_prev, mvms.squeeze(-1)).sum(dim=-1)
                    new_dkd = torch.mul(curr_conjugate_vec.squeeze(-1), mvms.squeeze(-1)).sum(dim=-1)
                    scale = torch.sqrt(torch.mul(dkd.abs(), new_dkd.abs().clamp_min(eps)))
                    rel_inner_products = torch.div(inner_products.abs(), scale.clamp_min(eps))

                    if not torch.sum(rel_inner_products.abs() > tolerance):
                        could_reorthogonalize = True
                        break
                
                # If reorthogonalization fails, stop storing and return the stored prefix
                if not could_reorthogonalize:
                    save_directions_cg = False
                    num_stored = k
            
            # Store the reorthogonalized direction and matching matrix-vector product
            d_mat[k].copy_(curr_conjugate_vec.squeeze(-1))
            kd_mat[k].copy_(mvms.squeeze(-1))
            num_stored = k + 1

        if precond:
            torch.mul(curr_conjugate_vec, mvms, out=mul_storage)
            torch.sum(mul_storage, -2, keepdim=True, out=alpha)

            # Do a safe division here
            torch.lt(alpha, eps, out=is_zero)
            alpha.masked_fill_(is_zero, 1)
            torch.div(residual_inner_prod, alpha, out=alpha)
            alpha.masked_fill_(is_zero, 0)

            # We'll cancel out any updates by setting alpha=0 for any vector that has already converged
            alpha.masked_fill_(has_converged, 0)

            # Update residual
            # residual_{k} = residual_{k-1} - alpha_{k} mat p_vec_{k-1}
            residual = torch.addcmul(residual, alpha, mvms, value=-1, out=residual)

            # Update precond_residual
            # precon_residual{k} = M^-1 residual_{k}
            precond_residual = preconditioner(residual)

            _jit_linear_cg_updates(
                result,
                alpha,
                residual_inner_prod,
                eps,
                beta,
                residual,
                precond_residual,
                mul_storage,
                is_zero,
                curr_conjugate_vec,
            )
        else:
            if save_directions_cg:
                _jit_linear_cg_updates_save_directions(
                    mvms,
                    result,
                    has_converged,
                    alpha,
                    residual_inner_prod,
                    eps,
                    beta,
                    residual,
                    precond_residual,
                    mul_storage,
                    is_zero,
                    curr_conjugate_vec,
                )
            else:
                _jit_linear_cg_updates_no_precond(
                    mvms,
                    result,
                    has_converged,
                    alpha,
                    residual_inner_prod,
                    eps,
                    beta,
                    residual,
                    precond_residual,
                    mul_storage,
                    is_zero,
                    curr_conjugate_vec,
                )

        torch.linalg.vector_norm(residual, ord=2, dim=-2, keepdim=True, out=residual_norm)
        residual_norm.masked_fill_(rhs_is_zero, 0)
        torch.lt(residual_norm, stop_updating_after, out=has_converged)
        
        if (
            k >= min(10, max_iter - 1)
            and bool(residual_norm.mean() < tolerance)
            and not (n_tridiag and k < min(n_tridiag_iter, max_iter - 1))
        ):
            tolerance_reached = True
            break

        # Update tridiagonal matrices, if applicable
        if n_tridiag and k < n_tridiag_iter and update_tridiag:
            alpha_tridiag = alpha.squeeze(-2).narrow(-1, 0, n_tridiag)
            beta_tridiag = beta.squeeze(-2).narrow(-1, 0, n_tridiag)
            torch.eq(alpha_tridiag, 0, out=alpha_tridiag_is_zero)
            alpha_tridiag.masked_fill_(alpha_tridiag_is_zero, 1)
            torch.reciprocal(alpha_tridiag, out=alpha_reciprocal)
            alpha_tridiag.masked_fill_(alpha_tridiag_is_zero, 0)

            if k == 0:
                t_mat[k, k].copy_(alpha_reciprocal)
            else:
                torch.addcmul(alpha_reciprocal, prev_beta, prev_alpha_reciprocal, out=t_mat[k, k])
                torch.mul(prev_beta.sqrt_(), prev_alpha_reciprocal, out=t_mat[k, k - 1])
                t_mat[k - 1, k].copy_(t_mat[k, k - 1])

                if t_mat[k - 1, k].max() < 1e-6:
                    update_tridiag = False

            last_tridiag_iter = k

            prev_alpha_reciprocal.copy_(alpha_reciprocal)
            prev_beta.copy_(beta_tridiag)

    # Un-normalize
    result = result.mul(rhs_norm)

    if not tolerance_reached and n_iter > 0:
        warnings.warn(
            "CG terminated in {} iterations with average residual norm {}"
            " which is larger than the tolerance of {} specified by"
            " linear_operator.settings.cg_tolerance."
            " If performance is affected, consider raising the maximum number of CG iterations by running code in"
            " a linear_operator.settings.max_cg_iterations(value) context.".format(
                k + 1, residual_norm.mean(), tolerance
            ),
        )

    if is_vector:
        result = result.squeeze(-1)

    if save_directions:
        d_mat = d_mat[:num_stored].permute(*range(1, 1 + len(batch_shape)), -1, 0).contiguous()
        return result, d_mat

    if n_tridiag:
        t_mat = t_mat[: last_tridiag_iter + 1, : last_tridiag_iter + 1]
        return (
            result,
            t_mat.permute(-1, *range(2, 2 + len(batch_shape)), 0, 1).contiguous(),
        )
    else:
        return result



def cg_store_lanczos_basis(
    matmul_closure,
    rhs,
    n_tridiag=0,
    tolerance=1e-10,
    eps=1e-10,
    stop_updating_after=1e-10,
    max_iter=100,
    max_tridiag_iter=0,
    initial_guess=None,
    preconditioner=None,
):
    """Run CG and recover an orthonormal basis from the stored directions.

    This wrapper calls ``linear_cg`` with ``save_directions=True``. During the
    CG solve, the modified routine stores reorthogonalized CG search directions
    in a matrix ``D``. A reduced QR factorization

        D = Q R

    is then used to convert these stored directions into an orthonormal basis
    ``Q`` for the same Krylov subspace.

    In this version, only the basis is returned. The projected matrix is not
    recovered from the triangular factor. Instead, downstream code constructs it
    directly as

        T = Q.T @ A @ Q,

    where multiplication by ``A`` is represented by ``matmul_closure``.

    Args:
        matmul_closure: Tensor or callable implementing multiplication by the
            system matrix.
        rhs: Right-hand side tensor passed to ``linear_cg``.
        n_tridiag: Passed through to ``linear_cg``. In the current prototype,
            tridiagonalization is not used by this wrapper.
        tolerance: CG convergence and reorthogonalization tolerance.
        eps: Small value used to avoid division by zero.
        stop_updating_after: Residual norm below which a right-hand side is
            treated as converged.
        max_iter: Maximum number of CG iterations.
        max_tridiag_iter: Passed through to ``linear_cg``.
        initial_guess: Optional initial guess for the CG solve.
        preconditioner: Optional preconditioner. Currently unsupported for this
            prototype wrapper.

    Returns:
        A tuple ``(result, q_mat)``, where ``result`` is the CG solution and
        ``q_mat`` is the QR-orthogonalized basis with shape ``[1, n, J]``.
    """
    
    if preconditioner is not None:
        raise NotImplementedError(
            "cg_store_lanczos_basis currently supports only the unpreconditioned case."
        )

    result, d_mat = linear_cg(
        matmul_closure,
        rhs,
        n_tridiag=n_tridiag,
        tolerance=tolerance,
        eps=eps,
        stop_updating_after=stop_updating_after,
        max_iter=max_iter,
        max_tridiag_iter=max_tridiag_iter,
        initial_guess=initial_guess,
        preconditioner=None,
        save_directions=True,
    )

    if d_mat.dim() != 3 or d_mat.shape[0] != 1:
        raise ValueError("This prototype currently expects d_mat with shape [1, n, J].")

    q_mat, _ = torch.linalg.qr(d_mat, mode="reduced")

    return result, q_mat












#!/usr/bin/env python3
from __future__ import annotations

import torch


def lanczos_tridiag(
    matmul_closure,
    max_iter,
    dtype,
    device,
    matrix_shape,
    batch_shape=torch.Size(),
    init_vecs=None,
    num_init_vecs=1,
    tol=1e-5,
):
    """ """
    # Determine batch mode
    multiple_init_vecs = False

    if not callable(matmul_closure):
        raise RuntimeError(
            "matmul_closure should be a function callable object that multiples a (Lazy)Tensor "
            "by a vector. Got a {} instead.".format(matmul_closure.__class__.__name__)
        )

    # Get initial probe ectors - and define if not available
    if init_vecs is None:
        init_vecs = torch.randn(matrix_shape[-1], num_init_vecs, dtype=dtype, device=device)
        init_vecs = init_vecs.expand(*batch_shape, matrix_shape[-1], num_init_vecs)

    else:
        init_vecs = init_vecs.to(dtype=dtype, device=device)
        num_init_vecs = init_vecs.size(-1)
        multiple_init_vecs = num_init_vecs > 1

        num_init_vecs = init_vecs.size(-1)

    # Define some constants
    num_iter = min(max_iter, matrix_shape[-1])
    dim_dimension = -2

    # Create storage for q_mat, alpha,and beta
    # q_mat - batch version of Q - orthogonal matrix of decomp
    # alpha - batch version main diagonal of T
    # beta - batch version of off diagonal of T
    q_mat = torch.zeros(
        num_iter,
        *batch_shape,
        matrix_shape[-1],
        num_init_vecs,
        dtype=dtype,
        device=device,
    )
    t_mat = torch.zeros(num_iter, num_iter, *batch_shape, num_init_vecs, dtype=dtype, device=device)

    # Begin algorithm
    # Initial Q vector: q_0_vec
    q_0_vec = init_vecs / torch.linalg.vector_norm(init_vecs, ord=2, dim=dim_dimension).unsqueeze(dim_dimension)
    q_mat[0].copy_(q_0_vec)

    # Initial alpha value: alpha_0
    r_vec = matmul_closure(q_0_vec)
    alpha_0 = q_0_vec.mul(r_vec).sum(dim_dimension)

    # Initial beta value: beta_0
    r_vec.sub_(alpha_0.unsqueeze(dim_dimension).mul(q_0_vec))
    beta_0 = torch.linalg.vector_norm(r_vec, ord=2, dim=dim_dimension)

    # Copy over alpha_0 and beta_0 to t_mat
    t_mat[0, 0].copy_(alpha_0)
    t_mat[0, 1].copy_(beta_0)
    t_mat[1, 0].copy_(beta_0)

    # Compute the first new vector
    q_mat[1].copy_(r_vec.div_(beta_0.unsqueeze(dim_dimension)))

    # Now we start the iteration
    for k in range(1, num_iter):
        # Get previous values
        q_prev_vec = q_mat[k - 1]
        q_curr_vec = q_mat[k]
        beta_prev = t_mat[k, k - 1].unsqueeze(dim_dimension)

        # Compute next alpha value
        r_vec = matmul_closure(q_curr_vec) - q_prev_vec.mul(beta_prev)
        alpha_curr = q_curr_vec.mul(r_vec).sum(dim_dimension, keepdim=True)
        # Copy over to t_mat
        t_mat[k, k].copy_(alpha_curr.squeeze(dim_dimension))

        # Copy over alpha_curr, beta_curr to t_mat
        if (k + 1) < num_iter:
            # Compute next residual value
            r_vec.sub_(alpha_curr.mul(q_curr_vec))
            # Full reorthogonalization: r <- r - Q (Q^T r)
            correction = r_vec.unsqueeze(0).mul(q_mat[: k + 1]).sum(dim_dimension, keepdim=True)
            correction = q_mat[: k + 1].mul(correction).sum(0)
            r_vec.sub_(correction)
            r_vec_norm = torch.linalg.vector_norm(r_vec, ord=2, dim=dim_dimension, keepdim=True)
            r_vec.div_(r_vec_norm)

            # Get next beta value
            beta_curr = r_vec_norm.squeeze_(dim_dimension)
            # Update t_mat with new beta value
            t_mat[k, k + 1].copy_(beta_curr)
            t_mat[k + 1, k].copy_(beta_curr)

            # Run more reorthoganilzation if necessary
            inner_products = q_mat[: k + 1].mul(r_vec.unsqueeze(0)).sum(dim_dimension)
            could_reorthogonalize = False
            for _ in range(10):
                if not torch.sum(inner_products > tol):
                    could_reorthogonalize = True
                    break
                correction = r_vec.unsqueeze(0).mul(q_mat[: k + 1]).sum(dim_dimension, keepdim=True)
                correction = q_mat[: k + 1].mul(correction).sum(0)
                r_vec.sub_(correction)
                r_vec_norm = torch.linalg.vector_norm(r_vec, ord=2, dim=dim_dimension, keepdim=True)
                r_vec.div_(r_vec_norm)
                inner_products = q_mat[: k + 1].mul(r_vec.unsqueeze(0)).sum(dim_dimension)

            # Update q_mat with new q value
            q_mat[k + 1].copy_(r_vec)

            if torch.sum(beta_curr.abs() > tol) == 0 or not could_reorthogonalize:
                break

    # Now let's transpose q_mat, t_mat intot the correct shape
    num_iter = k + 1

    # num_init_vecs x batch_shape x matrix_shape[-1] x num_iter
    q_mat = q_mat[:num_iter].permute(-1, *range(1, 1 + len(batch_shape)), -2, 0).contiguous()
    # num_init_vecs x batch_shape x num_iter x num_iter
    t_mat = t_mat[:num_iter, :num_iter].permute(-1, *range(2, 2 + len(batch_shape)), 0, 1).contiguous()

    # If we weren't in batch mode, remove batch dimension
    if not multiple_init_vecs:
        q_mat.squeeze_(0)
        t_mat.squeeze_(0)

    # We're done!
    return q_mat, t_mat



def extend_lanczos_basis_np(
    matmul_closure,
    max_iter,
    dtype,
    device,
    matrix_shape,
    q_mat,
    t_mat,
    tol=1e-12,
):
    if not callable(matmul_closure):
        raise RuntimeError(
            "matmul_closure should be a callable object that multiplies by a matrix."
        )

    q_mat = q_mat.to(dtype=dtype, device=device)
    t_mat = t_mat.to(dtype=dtype, device=device)

    if q_mat.dim() != 3:
        raise ValueError("This prototype expects q_mat with shape [batch, n, J].")

    if t_mat.dim() != 3:
        raise ValueError("This prototype expects t_mat with shape [batch, J, J].")

    batch_shape = q_mat.shape[:-2]
    num_rows = q_mat.size(-2)
    current_iter = q_mat.size(-1)
    dim_dimension = -2

    if matrix_shape[-1] != num_rows:
        raise ValueError("matrix_shape and q_mat do not agree.")

    if current_iter == 0:
        raise ValueError("q_mat must contain at least one basis vector.")

    if t_mat.shape != (*batch_shape, current_iter, current_iter):
        raise ValueError("t_mat must have shape [batch, J, J].")

    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")

    if max_iter <= current_iter:
        q_final = q_mat[..., :, :max_iter].contiguous()
        t_final = t_mat[..., :max_iter, :max_iter].contiguous()
        return q_final, 0.5 * (t_final + t_final.transpose(-1, -2))

    q_ext = q_mat.new_zeros(*batch_shape, num_rows, max_iter)
    q_ext[..., :, :current_iter] = q_mat

    t_ext = t_mat.new_zeros(*batch_shape, max_iter, max_iter)
    t_ext[..., :current_iter, :current_iter] = 0.5 * (
        t_mat + t_mat.transpose(-1, -2)
    )

    q = q_ext[..., :, current_iter - 1 : current_iter]

    v = matmul_closure(q)

    if v.shape != q.shape:
        raise ValueError("matmul_closure must return a tensor with the same shape as the basis vectors.")

    alpha_last = t_ext[..., current_iter - 1, current_iter - 1]
    v = v - alpha_last.unsqueeze(-1).unsqueeze(-1) * q

    if current_iter > 1:
        beta_prev = t_ext[..., current_iter - 2, current_iter - 1]
        q_prev = q_ext[..., :, current_iter - 2 : current_iter - 1]
        v = v - beta_prev.unsqueeze(-1).unsqueeze(-1) * q_prev

    for _ in range(2):
        q_prev_all = q_ext[..., :, :current_iter]
        coeffs = torch.matmul(q_prev_all.transpose(-1, -2), v)
        v = v - torch.matmul(q_prev_all, coeffs)

    num_iter = current_iter

    for j in range(current_iter, max_iter):
        beta = torch.linalg.vector_norm(v, ord=2, dim=dim_dimension, keepdim=True)

        if torch.sum(beta.abs() > tol) == 0:
            break

        beta_value = beta.squeeze(-1).squeeze(-1)

        t_ext[..., j - 1, j] = beta_value
        t_ext[..., j, j - 1] = beta_value

        q_prev = q
        q = v / beta

        q_ext[..., :, j : j + 1] = q
        num_iter = j + 1

        v = matmul_closure(q)

        if v.shape != q.shape:
            raise ValueError("matmul_closure must return a tensor with the same shape as the basis vectors.")

        alpha = torch.sum(q * v, dim=dim_dimension, keepdim=True)
        alpha_value = alpha.squeeze(-1).squeeze(-1)
        t_ext[..., j, j] = alpha_value

        v = v - alpha * q
        v = v - beta * q_prev

        for _ in range(2):
            q_prev_all = q_ext[..., :, : j + 1]
            coeffs = torch.matmul(q_prev_all.transpose(-1, -2), v)
            v = v - torch.matmul(q_prev_all, coeffs)

    q_final = q_ext[..., :, :num_iter].contiguous()
    t_final = t_ext[..., :num_iter, :num_iter].contiguous()

    return q_final, 0.5 * (t_final + t_final.transpose(-1, -2))



def extend_lanczos_basis_old(
    matmul_closure,
    max_iter,
    dtype,
    device,
    matrix_shape,
    q_mat,
    t_mat,
    tol=1e-12,
):
    if not callable(matmul_closure):
        raise RuntimeError(
            "matmul_closure should be a callable object that multiplies by a matrix."
        )

    q_mat = q_mat.to(dtype=dtype, device=device)
    t_mat = t_mat.to(dtype=dtype, device=device)

    if q_mat.dim() != 3:
        raise ValueError("This prototype expects q_mat with shape [batch, n, J].")

    if t_mat.dim() != 3:
        raise ValueError("This prototype expects t_mat with shape [batch, J, J].")

    batch_shape = q_mat.shape[:-2]
    num_rows = q_mat.size(-2)
    current_iter = q_mat.size(-1)
    target_iter = min(max_iter, matrix_shape[-1])
    dim_dimension = -2

    if matrix_shape[-1] != num_rows:
        raise ValueError("matrix_shape and q_mat do not agree.")

    if current_iter == 0:
        raise ValueError("q_mat must contain at least one basis vector.")

    if t_mat.shape != (*batch_shape, current_iter, current_iter):
        raise ValueError("t_mat must have shape [batch, J, J].")

    if target_iter <= 0:
        raise ValueError("max_iter must be positive.")

    if target_iter <= current_iter:
        q_final = q_mat[..., :, :target_iter].contiguous()
        t_final = t_mat[..., :target_iter, :target_iter].contiguous()
        return q_final, 0.5 * (t_final + t_final.transpose(-1, -2))

    q_ext = q_mat.new_zeros(*batch_shape, num_rows, target_iter)
    q_ext[..., :, :current_iter] = q_mat

    t_ext = t_mat.new_zeros(*batch_shape, target_iter, target_iter)
    t_ext[..., :current_iter, :current_iter] = 0.5 * (
        t_mat + t_mat.transpose(-1, -2)
    )

    q = q_ext[..., :, current_iter - 1 : current_iter]

    r_vec = matmul_closure(q)

    if r_vec.shape != q.shape:
        raise ValueError("matmul_closure must return a tensor with the same shape as the basis vectors.")

    alpha_last = t_ext[..., current_iter - 1, current_iter - 1]
    r_vec.sub_(q.mul(alpha_last.unsqueeze(-1).unsqueeze(-1)))

    if current_iter > 1:
        beta_prev = t_ext[..., current_iter - 2, current_iter - 1]
        q_prev = q_ext[..., :, current_iter - 2 : current_iter - 1]
        r_vec.sub_(q_prev.mul(beta_prev.unsqueeze(-1).unsqueeze(-1)))

    for _ in range(2):
        q_prev_all = q_ext[..., :, :current_iter]
        coeffs = torch.matmul(q_prev_all.transpose(-1, -2), r_vec)
        r_vec.sub_(torch.matmul(q_prev_all, coeffs))
    
    num_iter = current_iter

    for k in range(current_iter, target_iter):
        beta = torch.linalg.vector_norm(r_vec, ord=2, dim=dim_dimension, keepdim=True)

        if torch.sum(beta.abs() > tol) == 0:
            break

        beta_value = beta.squeeze(-1).squeeze(-1)

        t_ext[..., k - 1, k] = beta_value
        t_ext[..., k, k - 1] = beta_value

        q_prev = q
        q = r_vec.div(beta)

        q_ext[..., :, k : k + 1] = q
        num_iter += 1

        r_vec = matmul_closure(q)

        alpha = torch.sum(q * r_vec, dim=dim_dimension, keepdim=True)
        alpha_value = alpha.squeeze(-1).squeeze(-1)
        t_ext[..., k, k] = alpha_value

        r_vec.sub_(q.mul(alpha))
        r_vec.sub_(q_prev.mul(beta))

        q_prev_all = q_ext[..., :, : k + 1]

        correction = r_vec.mul(q_prev_all).sum(dim_dimension, keepdim=True)
        correction = q_prev_all.mul(correction).sum(-1, keepdim=True)
        r_vec.sub_(correction)

        r_vec_norm = torch.linalg.vector_norm(r_vec, ord=2, dim=dim_dimension, keepdim=True)

        inner_products = q_prev_all.mul(r_vec.div(r_vec_norm)).sum(dim_dimension)

        could_reorthogonalize = False
        for _ in range(10):
            if not torch.sum(inner_products.abs() > tol):
                could_reorthogonalize = True
                break

            correction = r_vec.mul(q_prev_all).sum(dim_dimension, keepdim=True)
            correction = q_prev_all.mul(correction).sum(-1, keepdim=True)
            r_vec.sub_(correction)

            r_vec_norm = torch.linalg.vector_norm(r_vec, ord=2, dim=dim_dimension, keepdim=True)

            inner_products = q_prev_all.mul(r_vec.div(r_vec_norm)).sum(dim_dimension)

        if not could_reorthogonalize:
            break

    q_final = q_ext[..., :, :num_iter].contiguous()
    t_final = t_ext[..., :num_iter, :num_iter].contiguous()

    return q_final, 0.5 * (t_final + t_final.transpose(-1, -2))


def extend_lanczos_basis_np(
    matmul_closure,
    max_iter,
    dtype,
    device,
    matrix_shape,
    q_mat,
    t_mat,
    tol=1e-6,
):
    if not callable(matmul_closure):
        raise RuntimeError(
            "matmul_closure should be a callable object that multiplies by a matrix."
        )

    q_mat = q_mat.to(dtype=dtype, device=device)
    t_mat = t_mat.to(dtype=dtype, device=device)

    if q_mat.dim() != 3:
        raise ValueError("This prototype expects q_mat with shape [batch, n, J].")

    if t_mat.dim() != 3:
        raise ValueError("This prototype expects t_mat with shape [batch, J, J].")

    batch_shape = q_mat.shape[:-2]
    num_rows = q_mat.size(-2)
    current_iter = q_mat.size(-1)
    target_iter = min(max_iter, matrix_shape[-1])
    dim_dimension = -2

    if matrix_shape[-1] != num_rows:
        raise ValueError("matrix_shape and q_mat do not agree.")

    if current_iter == 0:
        raise ValueError("q_mat must contain at least one basis vector.")

    if t_mat.shape != (*batch_shape, current_iter, current_iter):
        raise ValueError("t_mat must have shape [batch, J, J].")

    if target_iter <= 0:
        raise ValueError("max_iter must be positive.")

    if target_iter <= current_iter:
        q_final = q_mat[..., :, :target_iter].contiguous()
        t_final = t_mat[..., :target_iter, :target_iter].contiguous()
        return q_final, 0.5 * (t_final + t_final.transpose(-1, -2))

    q_ext = q_mat.new_zeros(target_iter, *batch_shape, num_rows, 1)
    q_ext[:current_iter].copy_(q_mat.permute(-1, *range(len(batch_shape)), -2).unsqueeze(-1))

    t_ext = t_mat.new_zeros(target_iter, target_iter, *batch_shape, 1)
    t_ext[:current_iter, :current_iter].copy_(t_mat.permute(-2, -1, *range(len(batch_shape))).unsqueeze(-1))

    q_prev_vec = q_ext[current_iter - 2]
    q_curr_vec = q_ext[current_iter - 1]
    beta_prev = t_ext[current_iter - 1, current_iter - 2].unsqueeze(dim_dimension)
    
    r_vec = matmul_closure(q_curr_vec) - q_prev_vec.mul(beta_prev)

    alpha_curr = q_curr_vec.mul(r_vec).sum(dim_dimension, keepdim=True)
    t_ext[current_iter - 1, current_iter - 1].copy_(alpha_curr.squeeze(dim_dimension))

    r_vec.sub_(alpha_curr.mul(q_curr_vec))

    q_prev_all = q_ext[: current_iter]

    correction = r_vec.unsqueeze(0).mul(q_prev_all).sum(dim=dim_dimension,keepdim=True)
    correction = q_prev_all.mul(correction).sum(0)
    r_vec.sub_(correction)

    r_vec_norm = torch.linalg.vector_norm(r_vec, ord=2, dim=dim_dimension, keepdim=True)

    r_vec.div_(r_vec_norm)
    inner_products = q_prev_all.mul(r_vec.unsqueeze(0)).sum(dim_dimension)

    could_reorthogonalize = False

    for _ in range(10):
        if not torch.sum(inner_products.abs() > tol):
            could_reorthogonalize = True
            break

        correction = r_vec.unsqueeze(0).mul(q_prev_all).sum(dim=dim_dimension,keepdim=True)
        correction = q_prev_all.mul(correction).sum(0)
        r_vec.sub_(correction)

        r_vec_norm = torch.linalg.vector_norm(r_vec, ord=2, dim=dim_dimension, keepdim=True)

        r_vec.div_(r_vec_norm)
        inner_products = q_prev_all.mul(r_vec.unsqueeze(0)).sum(dim_dimension)

    q_ext[current_iter].copy_(r_vec)

    if not could_reorthogonalize:
        target_iter = current_iter
    
    num_iter = current_iter + 1

    for k in range(current_iter, target_iter):
        q_prev_vec = q_ext[k - 1]
        q_curr_vec = q_ext[k]
        beta_prev = t_ext[k, k - 1].unsqueeze(dim_dimension)

        r_vec = matmul_closure(q_curr_vec) - q_prev_vec.mul(beta_prev)
        alpha_curr = q_curr_vec.mul(r_vec).sum(dim_dimension, keepdim=True)

        t_ext[k, k].copy_(alpha_curr.squeeze(dim_dimension))

        if (k + 1) < target_iter:
            r_vec.sub_(alpha_curr.mul(q_curr_vec))

            q_prev_all = q_ext[: k + 1]

            correction = r_vec.unsqueeze(0).mul(q_prev_all).sum(dim_dimension, keepdim=True)
            correction = q_prev_all.mul(correction).sum(0)
            r_vec.sub_(correction)

            r_vec_norm = torch.linalg.vector_norm(r_vec, ord=2, dim=dim_dimension, keepdim=True)
            beta_curr = r_vec_norm.squeeze(dim_dimension)

            if torch.sum(beta_curr.abs() > tol) == 0:
                break

            t_ext[k, k + 1].copy_(beta_curr)
            t_ext[k + 1, k].copy_(beta_curr)

            r_vec.div_(r_vec_norm)

            inner_products = q_prev_all.mul(r_vec.unsqueeze(0)).sum(dim_dimension)

            could_reorthogonalize = False
            for _ in range(10):
                if not torch.sum(inner_products.abs() > tol):
                    could_reorthogonalize = True
                    break

                correction = r_vec.unsqueeze(0).mul(q_prev_all).sum(dim_dimension, keepdim=True)
                correction = q_prev_all.mul(correction).sum(0)
                r_vec.sub_(correction)

                r_vec_norm = torch.linalg.vector_norm(r_vec, ord=2, dim=dim_dimension, keepdim=True)
                if torch.sum(r_vec_norm.abs() > tol) == 0:
                    break

                r_vec.div_(r_vec_norm)
                inner_products = q_prev_all.mul(r_vec.unsqueeze(0)).sum(dim_dimension)

            if not could_reorthogonalize:
                break

            q_ext[k + 1].copy_(r_vec)
            num_iter = k + 2

    q_final = q_ext[:num_iter].squeeze(-1).permute(*range(1, 1 + len(batch_shape)), -1, 0).contiguous()
    t_final = t_ext[:num_iter, :num_iter].squeeze(-1).permute(*range(2, 2 + len(batch_shape)), 0, 1).contiguous()

    #t_final = torch.matmul(q_final.squeeze(-1).transpose(-1,-2), matmul_closure(q_final.squeeze(-1)))

    return q_final, 0.5 * (t_final + t_final.transpose(-1, -2))


def extend_lanczos_basis(
    matmul_closure,
    max_iter,
    dtype,
    device,
    matrix_shape,
    q_mat,
    tol=1e-6,
):
    if not callable(matmul_closure):
        raise RuntimeError(
            "matmul_closure should be a callable object that multiplies by a matrix."
        )

    q_mat = q_mat.to(dtype=dtype, device=device)

    if q_mat.dim() != 3:
        raise ValueError("This prototype expects q_mat with shape [batch, n, J].")

    batch_shape = q_mat.shape[:-2]
    num_rows = q_mat.size(-2)
    current_iter = q_mat.size(-1)
    target_iter = min(max_iter, matrix_shape[-1])
    dim_dimension = -2

    if matrix_shape[-1] != num_rows:
        raise ValueError("matrix_shape and q_mat do not agree.")

    if current_iter == 0:
        raise ValueError("q_mat must contain at least one basis vector.")
    
    if target_iter <= 0:
        raise ValueError("max_iter must be positive.")

    if target_iter <= current_iter:
        q_final = q_mat[..., :, :target_iter].contiguous()
        t_final = torch.matmul(q_final.transpose(-1,-2), matmul_closure(q_final))
        return q_final, 0.5 * (t_final + t_final.transpose(-1, -2))

    q_ext = q_mat.new_zeros(target_iter, *batch_shape, num_rows)
    q_ext[:current_iter].copy_(q_mat.permute(-1, *range(len(batch_shape)), -2))

    q = q_ext[current_iter - 1].unsqueeze(-1)

    r_vec = matmul_closure(q)

    if r_vec.shape != q.shape:
        raise ValueError("matmul_closure must return a tensor with the same shape as the basis vectors.")
    Kq_last = matmul_closure(q_mat[:, :, current_iter - 1 : current_iter])
    alpha_last = torch.matmul(q_mat[:, :, current_iter - 1 : current_iter].transpose(-1,-2), Kq_last)
    r_vec.sub_(q.mul(alpha_last))

    if current_iter > 1:
        beta_prev = torch.matmul(q_mat[:, :, current_iter - 2 : current_iter - 1].transpose(-1,-2), Kq_last)
        q_prev = q_ext[current_iter - 2].unsqueeze(-1)
        r_vec.sub_(q_prev.mul(beta_prev))

    q_prev_all = q_ext[: current_iter]

    correction = r_vec.unsqueeze(0).mul(q_prev_all.unsqueeze(-1)).sum(dim=dim_dimension,keepdim=True)
    correction = q_prev_all.unsqueeze(-1).mul(correction).sum(0)
    r_vec.sub_(correction)

    r_vec_norm = torch.linalg.vector_norm(r_vec, ord=2, dim=dim_dimension, keepdim=True)

    r_vec.div_(r_vec_norm)
    inner_products = q_prev_all.unsqueeze(-1).mul(r_vec.unsqueeze(0)).sum(dim_dimension)

    could_reorthogonalize = False

    for _ in range(10):
        if not torch.sum(inner_products.abs() > tol):
            could_reorthogonalize = True
            break

        correction = r_vec.unsqueeze(0).mul(q_prev_all.unsqueeze(-1)).sum(dim=dim_dimension,keepdim=True)
        correction = q_prev_all.unsqueeze(-1).mul(correction).sum(0)
        r_vec.sub_(correction)

        r_vec_norm = torch.linalg.vector_norm(r_vec, ord=2, dim=dim_dimension, keepdim=True)

        r_vec.div_(r_vec_norm)
        inner_products = q_prev_all.unsqueeze(-1).mul(r_vec.unsqueeze(0)).sum(dim_dimension)

    if not could_reorthogonalize:
        target_iter = current_iter
    
    num_iter = current_iter

    for k in range(current_iter, target_iter):
        beta = torch.linalg.vector_norm(r_vec, ord=2, dim=dim_dimension, keepdim=True)

        if torch.sum(beta.abs() > tol) == 0:
            break
        q_prev = q
        q = r_vec.div(beta)

        q_ext[k].copy_(q.squeeze(-1))
        num_iter += 1

        r_vec = matmul_closure(q)

        alpha = torch.sum(q * r_vec, dim=dim_dimension, keepdim=True)

        r_vec.sub_(q.mul(alpha))
        r_vec.sub_(q_prev.mul(beta))

        q_prev_all = q_ext[: k + 1]

        correction = r_vec.unsqueeze(0).mul(q_prev_all.unsqueeze(-1)).sum(dim=dim_dimension,keepdim=True)
        correction = q_prev_all.unsqueeze(-1).mul(correction).sum(0)
        r_vec.sub_(correction)

        r_vec_norm = torch.linalg.vector_norm(r_vec, ord=2, dim=dim_dimension, keepdim=True)

        inner_products = q_prev_all.unsqueeze(-1).mul(r_vec.div(r_vec_norm).unsqueeze(0)).sum(dim_dimension)

        could_reorthogonalize = False

        for _ in range(10):
            if not torch.sum(inner_products.abs() > tol):
                could_reorthogonalize = True
                break

            correction = r_vec.unsqueeze(0).mul(q_prev_all.unsqueeze(-1)).sum(dim=dim_dimension,keepdim=True)
            correction = q_prev_all.unsqueeze(-1).mul(correction).sum(0)
            r_vec.sub_(correction)

            r_vec_norm = torch.linalg.vector_norm(r_vec, ord=2, dim=dim_dimension, keepdim=True)

            inner_products = q_prev_all.unsqueeze(-1).mul(r_vec.div(r_vec_norm).unsqueeze(0)).sum(dim_dimension)


        if not could_reorthogonalize:
            break

    q_final = q_ext[:num_iter].permute(*range(1, 1 + len(batch_shape)), -1, 0).contiguous()

    t_final = torch.matmul(q_final.transpose(-1,-2), matmul_closure(q_final))

    return q_final, 0.5 * (t_final + t_final.transpose(-1, -2))
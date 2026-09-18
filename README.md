# CG Lanczos Variance Estimator

This repository contains NumPy and PyTorch prototype implementations for Gaussian process posterior variance approximation.

The main idea is to reuse Krylov subspace information generated during conjugate gradients (CG) for the posterior mean solve in order to improve or reduce the cost of Lanczos-based posterior covariance approximations.

## Methods

The repository currently includes:

- exact Gaussian process posterior covariance computation,
- LOVE-style Lanczos variance estimation,
- CG search-direction based covariance approximations,
- QR-stabilized CG covariance correction,
- CG-derived Lanczos basis recovery,
- Lanczos basis extension from stored CG information,
- PyTorch prototypes based on the structure of `linear_operator` / `gpytorch`.

## Current Main Idea

During the CG solve for

\[
\widehat K \alpha = y,
\]

the CG search directions span a Krylov subspace that can be reused for covariance approximation. Instead of starting LOVE from scratch, the prototype stores CG directions during the posterior mean solve.

The stored search directions are collected in a matrix \(D\). A reduced QR factorization

\[
D = QR
\]

is then used to recover an orthonormal basis \(Q\). The recovered basis can either be used directly or extended by additional Lanczos iterations.

In this `torch_v2` prototype, the projected matrix is not recovered from stored matrix-vector products. Instead, after the final basis \(Q_{\mathrm{ext}}\) has been constructed, the projected matrix is computed directly as

\[
T_{\mathrm{ext}}
=
Q_{\mathrm{ext}}^\top \widehat K Q_{\mathrm{ext}}.
\]

This gives a LOVE-like covariance approximation

\[
Q_{\mathrm{ext}} T_{\mathrm{ext}}^{-1} Q_{\mathrm{ext}}^\top
\approx
\widehat K^{-1},
\]

where the initial part of \(Q_{\mathrm{ext}}\) is recovered during CG.

## Torch Version

This branch corresponds to the `torch_v2` prototype.

In this version:

- CG directions are stored during the CG solve.
- The stored CG directions are reorthogonalized in the \(K\)-inner product.
- A reduced QR factorization is used to obtain an orthonormal basis.
- The basis is extended by additional Lanczos iterations if needed.
- The projected matrix is computed directly as \(T = Q^\top KQ\) after the final basis has been constructed.

This version is more robust than recovering \(KQ\) through a triangular solve, because the final projected matrix is computed directly from the final basis. The downside is that forming \(T = Q^\top KQ\) requires an additional matrix-matrix multiplication with the full covariance matrix. Therefore, the theoretical cost is closer to standard LOVE, although the operation can still be efficient in practice because it is performed as a matrix-matrix multiplication rather than as many sequential matrix-vector products.

## Structure

```text
src/
  kernels/        Kernel implementations
  means/          Mean functions
  linalg/         CG and Lanczos routines
  corrections.py  Covariance correction formulas
  diagnostics.py  Error and stability diagnostics
  gp.py           Minimal GP model

src_torch/
  kernels/        PyTorch kernel implementations
  means/          PyTorch mean functions
  linalg/         PyTorch CG and Lanczos prototype routines
  corrections.py  PyTorch covariance correction formulas
  diagnostics.py  PyTorch diagnostics
  gp.py           Minimal PyTorch GP model

experiments/
  NumPy experiments for visualizing posterior uncertainty and comparing accuracy and runtime.

experiments_torch/
  PyTorch experiments comparing CG-initialized Lanczos extension with standard LOVE.

tests/
  Unit tests for kernels, means, linear algebra, corrections, and GP predictions.
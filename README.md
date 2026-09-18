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

the CG search directions span a Krylov subspace that can be reused for covariance approximation. Instead of starting LOVE from scratch, the prototype stores CG directions and corresponding matrix-vector products during the posterior mean solve.

The stored search directions are collected in a matrix \(D\), and the corresponding products are collected in \(KD\). A reduced QR factorization

\[
D = QR
\]

is then used to recover an orthonormal basis \(Q\). In this `torch_v1` prototype, the associated matrix-vector products are recovered by solving

\[
R^\top (KQ)^\top = (KD)^\top.
\]

This gives the projected matrix

\[
T = Q^\top KQ.
\]

The recovered basis can then either be used directly or extended by additional Lanczos iterations. This gives a LOVE-like covariance approximation

\[
Q_{\mathrm{ext}} T_{\mathrm{ext}}^{-1} Q_{\mathrm{ext}}^\top
\approx
\widehat K^{-1},
\]

where the initial part of \(Q_{\mathrm{ext}}\) is recovered during CG.

## Torch Version

This branch corresponds to the `torch_v1` prototype.

In this version:

- CG directions and corresponding matrix-vector products are stored during the CG solve.
- The stored CG directions are reorthogonalized in the \(K\)-inner product.
- A reduced QR factorization is used to obtain an orthonormal basis.
- The products \(KQ\) are recovered from the stored \(KD\) values via a triangular solve.
- The projected matrix \(T = Q^\top KQ\) is formed from the recovered \(KQ\).
- The basis is then extended by additional Lanczos iterations if needed.

This version is computationally cheaper than explicitly recomputing \(KQ\), but it is also more sensitive to numerical stability. In particular, the quality of the recovered \(KQ\) depends on the conditioning of the QR factor and on the accuracy of the stored CG directions and products.

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
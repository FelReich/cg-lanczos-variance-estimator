# CG Lanczos Variance Estimator

This repository contains NumPy implementations and experiments for Gaussian process posterior variance approximation.

The main idea is to reuse Krylov subspace information generated during conjugate gradients (CG) for the posterior mean solve in order to improve or reduce the cost of Lanczos-based posterior covariance approximations.

## Methods

The repository currently includes:

- exact Gaussian process posterior covariance computation,
- LOVE-style Lanczos variance estimation,
- CG search-direction based covariance approximations,
- QR-stabilized CG covariance correction,
- CG-residual based Lanczos basis recovery,
- Lanczos basis extension from stored CG residual information.

## Current Main Idea

During the CG solve for

\[
\widehat K \alpha = y,
\]

the normalized CG residuals contain Lanczos basis information. Instead of starting LOVE from scratch, this basis can be stored and either used directly or extended with additional Lanczos steps.

This gives a LOVE-like covariance approximation

\[
Q_{\mathrm{ext}} T_{\mathrm{ext}}^{-1} Q_{\mathrm{ext}}^\top
\approx
\widehat K^{-1},
\]

where the initial part of \(Q_{\mathrm{ext}}\) is recovered during CG. If no additional Lanczos steps are used, then \(Q_{\mathrm{ext}}\) is simply the basis obtained from the CG residuals. Otherwise, the stored basis is continued by further Lanczos iterations.

## Structure

```text
src/
  kernels/        Kernel implementations
  means/          Mean functions
  linalg/         CG and Lanczos routines
  corrections.py  Covariance correction formulas
  diagnostics.py  Error and stability diagnostics
  gp.py           Minimal GP model

experiments/
  Scripts for visualizing posterior uncertainty and comparing the accuracy and runtime of CG-based approximations with LOVE.

tests/
  Unit tests for kernels, means, linear algebra, corrections, and GP predictions
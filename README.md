# CG Lanczos Variance Estimator

This repository documents the development of an approach to reusing conjugate-gradient (CG) information for LOVE-based Gaussian process posterior variance approximation, initially implemented in NumPy and subsequently in PyTorch.

The resulting approach has been integrated into customized versions of
[GPyTorch](https://github.com/FelReich/gpytorch) and
[linear_operator](https://github.com/FelReich/linear_operator).


## Research Proposal and Benchmarks

The theoretical proposal, experimental results, and KeOps GPU benchmark
script are available in [`benchmark_and_proposal/`](benchmark_and_proposal/):

- [`ext_love_theory.pdf`](benchmark_and_proposal/ext_love_theory.pdf): 
  theoretical derivations and experimental findings.
- [`run_keops_benchmark.py`](benchmark_and_proposal/run_keops_benchmark.py): 
  benchmark comparing the CG-Lanczos extension with standard LOVE.

### Reproducing the KeOps Benchmarks

To reproduce the results, place the benchmark script and the two customized libraries in the same parent directory:

```text
benchmark_workspace/
├── run_keops_benchmark.py
├── gpytorch/
└── linear_operator/
```

Clone the two libraries from
[GPyTorch](https://github.com/FelReich/gpytorch) and
[linear_operator](https://github.com/FelReich/linear_operator).

With the required dependencies installed, run:

```bash
python run_keops_benchmark.py --device cuda --dtype float64 --no-preconditioner
```

The script evaluates 200 matrix-free KeOps configurations with 100,000 training points, recording prediction times, GPU memory consumption, and numerical diagnostics in `keops_final_results.csv`.

## Methods

The repository includes:

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
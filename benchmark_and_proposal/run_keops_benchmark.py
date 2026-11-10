#!/usr/bin/env python3
"""Compare CG-Lanczos extension with LOVE on large matrix-free KeOps GPs.

The schedule has 120 distinct kernel/function/hyperparameter configurations,
72 paired 500-step limit studies, and eight paired 1000-step edge cases.
Only predictive means and marginal variances are evaluated; no exact reference
or dense training/test covariance is computed. Each case is flushed to CSV.
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
import warnings
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import torch


KERNELS = ("rbf", "matern32", "matern52", "periodic")
FUNCTIONS = ("smooth_mix", "periodic_mix", "rough_mix", "localized", "trend_mix")
REGIMES = (
    ("balanced", 0.3, 1.0, 1e-2),
    ("low_noise", 0.3, 1.0, 1e-4),
    ("long_length", 3.0, 1.0, 1e-2),
    ("short_length", 0.1, 1.0, 1e-4),
    ("low_scale", 1.0, 0.3, 1e-4),
    ("high_scale", 1.0, 3.0, 1e-2),
)


@dataclass(frozen=True)
class Case:
    case_id: int
    config_id: int
    tier: str
    kernel: str
    function: str
    regime: str
    lengthscale: float
    outputscale: float
    noise: float
    max_cg_iterations: int
    max_root_size: int


def make_cases() -> tuple[Case, ...]:
    configurations = []
    for regime, lengthscale, outputscale, noise in REGIMES:
        for function in FUNCTIONS:
            for kernel in KERNELS:
                configurations.append((kernel, function, regime, lengthscale, outputscale, noise))

    cases = []

    def add(config_id: int, tier: str, cg_limit: int, root_limit: int) -> None:
        cases.append(Case(len(cases), config_id, tier, *configurations[config_id], cg_limit, root_limit))

    for config_id in range(len(configurations)):
        add(config_id, "baseline", 300, 200)

    # The same 24 data configurations isolate CG, root, and joint limit changes.
    anchors = [
        config_id
        for config_id, (_, function, regime, *_rest) in enumerate(configurations)
        if regime == "low_noise" or (regime == "short_length" and function == "smooth_mix")
    ]
    for tier, cg_limit, root_limit in (
        ("cg500", 500, 200),
        ("root500", 300, 500),
        ("both500", 500, 500),
    ):
        for config_id in anchors:
            add(config_id, tier, cg_limit, root_limit)

    edge_configs = [
        config_id
        for config_id in anchors
        if configurations[config_id][2] == "short_length"
        or configurations[config_id][1] == "rough_mix"
    ]
    for config_id in edge_configs:
        add(config_id, "both1000", 1000, 1000)

    assert len(configurations) == 120 and len(anchors) == 24 and len(edge_configs) == 8
    assert len(cases) == 200
    return tuple(cases)


CASES = make_cases()
CASE_FIELDS = (
    "config_id", "tier", "kernel", "function", "regime", "lengthscale",
    "outputscale", "noise", "max_cg_iterations", "max_root_size",
)
RUN_FIELDS = (
    "n", "m", "dtype", "device", "seed", "cg_tolerance",
    "lanczos_tolerance", "no_preconditioner", "aggressive_mean_stop",
)
FIELDS = (
    "case_id", *CASE_FIELDS, *RUN_FIELDS, "order", "status", "J_resid",
    "J_ext", "J_love", "time_ext", "time_love", "rel_time",
    "peak_gib_ext", "peak_gib_love", "rel_mean", "max_abs_mean",
    "rel_variance", "max_abs_variance", "min_variance_ext",
    "min_variance_love", "cg_warning_ext", "cg_warning_love",
    "error_ext", "error_love",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100000)
    parser.add_argument("--m", type=int, default=10000)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cg-tolerance", type=float, default=1e-4)
    parser.add_argument("--lanczos-tolerance", type=float, help="Defaults to the CG tolerance")
    parser.add_argument("--no-preconditioner", action="store_true")
    parser.add_argument("--aggressive-mean-stop", action="store_true")
    parser.add_argument("--case-ids", help="Comma-separated case IDs from --dry-run")
    parser.add_argument("--max-cases", type=int, help="Run only the first N selected cases")
    parser.add_argument("--output", type=Path, default=Path("keops_final_results.csv"))
    parser.add_argument("--overwrite", action="store_true", help="Start a new CSV at --output")
    parser.add_argument("--dry-run", action="store_true", help="Print the selected case schedule only")
    return parser.parse_args()


def select_cases(args: argparse.Namespace) -> list[Case]:
    if args.case_ids:
        try:
            indices = [int(part.strip()) for part in args.case_ids.split(",")]
        except ValueError as exc:
            raise SystemExit("--case-ids must contain comma-separated integers") from exc
        if len(set(indices)) != len(indices) or any(index < 0 or index >= len(CASES) for index in indices):
            raise SystemExit(f"--case-ids must contain unique indices between 0 and {len(CASES) - 1}")
    else:
        indices = list(range(len(CASES)))
    if args.max_cases is not None:
        if args.max_cases < 1:
            raise SystemExit("--max-cases must be positive")
        indices = indices[:args.max_cases]
    return [CASES[index] for index in indices]


def target_function(name: str, x: torch.Tensor) -> torch.Tensor:
    z = x.squeeze(-1)
    if name == "smooth_mix":
        return torch.sin(z) + torch.cos(3 * z) + torch.exp(-(z**2))
    if name == "periodic_mix":
        return torch.sin(2 * z) + 0.3 * torch.sin(7 * z) + 0.1 * torch.cos(17 * z)
    if name == "rough_mix":
        return torch.sin(8 * z) + 0.25 * torch.cos(31 * z) + 0.1 * torch.sin(53 * z)
    if name == "localized":
        return torch.exp(-12 * (z - 0.75) ** 2) - 0.6 * torch.exp(-8 * (z + 1.5) ** 2)
    if name == "trend_mix":
        return 0.3 * z + torch.sin(2 * z) + 0.1 * z**2
    raise ValueError(name)


def relative_error(approx: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(reference).clamp_min(torch.finfo(reference.dtype).eps)
    return (torch.linalg.vector_norm(approx - reference) / denominator).item()


def run_metadata(args: argparse.Namespace) -> dict[str, str]:
    metadata = {field: str(getattr(args, field)) for field in RUN_FIELDS if field != "lanczos_tolerance"}
    metadata["lanczos_tolerance"] = str(
        args.cg_tolerance if args.lanczos_tolerance is None else args.lanczos_tolerance
    )
    return metadata


def case_fields(case: Case) -> dict[str, str]:
    return {field: str(getattr(case, field)) for field in CASE_FIELDS}


def completed_cases(path: Path, metadata: dict[str, str]) -> set[int]:
    if not path.exists():
        return set()
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != list(FIELDS):
            raise SystemExit(f"CSV columns differ from this script: {path}. Use another --output path.")
        rows = list(reader)
    for row in rows:
        case_id = int(row["case_id"])
        if case_id < 0 or case_id >= len(CASES):
            raise SystemExit(f"Unknown case ID {case_id} in {path}.")
        expected = {**case_fields(CASES[case_id]), **metadata}
        if any(row[field] != value for field, value in expected.items()):
            raise SystemExit(f"Case or run settings changed in {path}; use another --output path.")
    return {int(row["case_id"]) for row in rows}


def main() -> None:
    args = parse_args()
    cases = select_cases(args)
    counts = Counter(case.tier for case in CASES)
    print("Schedule:", len(CASES), "cases;", dict(counts), flush=True)
    if args.dry_run:
        for case in cases:
            print(
                f"{case.case_id:3d} config={case.config_id:3d} {case.tier:8s} "
                f"{case.kernel:8s} {case.function:12s} {case.regime:12s} "
                f"ls={case.lengthscale:g} scale={case.outputscale:g} noise={case.noise:.0e} "
                f"CG={case.max_cg_iterations} root={case.max_root_size}"
            )
        return

    if args.n < 2 or args.m < 1 or args.cg_tolerance <= 0:
        raise SystemExit("Require n >= 2, m >= 1, and positive CG tolerance.")
    if args.lanczos_tolerance is not None and args.lanczos_tolerance <= 0:
        raise SystemExit("Lanczos tolerance must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Use --device cpu only for a small smoke test.")
    try:
        from pykeops.torch import LazyTensor  # noqa: F401
    except ImportError as exc:
        raise SystemExit("PyKeOps is required in this Python environment.") from exc

    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "linear_operator"))
    sys.path.insert(0, str(root / "gpytorch"))
    import gpytorch
    import linear_operator
    from linear_operator.utils.warnings import NumericalWarning

    class ExactKernelGP(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood, covar_module):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = covar_module

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    lanczos_tolerance = args.cg_tolerance if args.lanczos_tolerance is None else args.lanczos_tolerance
    generator = torch.Generator(device=device).manual_seed(args.seed)
    train_x = -3 + 6 * torch.rand(args.n, 1, dtype=dtype, device=device, generator=generator)
    train_x = train_x.sort(dim=0).values
    test_x = torch.linspace(-3, 3, args.m, dtype=dtype, device=device).unsqueeze(-1)
    targets = {name: target_function(name, train_x) for name in FUNCTIONS}

    def make_model(case: Case, train_y: torch.Tensor):
        if case.kernel == "rbf":
            base = gpytorch.kernels.keops.RBFKernel()
        elif case.kernel == "matern32":
            base = gpytorch.kernels.keops.MaternKernel(nu=1.5)
        elif case.kernel == "matern52":
            base = gpytorch.kernels.keops.MaternKernel(nu=2.5)
        elif case.kernel == "periodic":
            base = gpytorch.kernels.keops.PeriodicKernel()
        else:
            raise ValueError(case.kernel)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-10)
        ).to(device=device, dtype=dtype)
        model = ExactKernelGP(train_x, train_y, likelihood, gpytorch.kernels.ScaleKernel(base)).to(
            device=device, dtype=dtype
        )
        model.covar_module.base_kernel.lengthscale = case.lengthscale
        if case.kernel == "periodic":
            model.covar_module.base_kernel.period_length = 1.0
        model.covar_module.outputscale = case.outputscale
        likelihood.noise = case.noise
        model.eval()
        likelihood.eval()
        return model

    def run_method(case: Case, train_y: torch.Tensor, method: str) -> dict:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        model = make_model(case, train_y)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always", NumericalWarning)
            with ExitStack() as stack:
                for context in (
                    torch.no_grad(),
                    gpytorch.settings.use_keops(True),
                    gpytorch.settings.fast_pred_var(True),
                    gpytorch.settings.fast_computations(solves=True),
                    gpytorch.settings.max_cholesky_size(2),
                    gpytorch.settings.max_cg_iterations(case.max_cg_iterations),
                    gpytorch.settings.max_root_decomposition_size(case.max_root_size),
                    gpytorch.settings.cg_tolerance(args.cg_tolerance),
                    gpytorch.settings.eval_cg_tolerance(lanczos_tolerance),
                    gpytorch.settings.use_cg_lanczos_variance(method == "ext"),
                    gpytorch.settings.cg_lanczos_aggressive_mean_stop(args.aggressive_mean_stop),
                ):
                    stack.enter_context(context)
                if args.no_preconditioner:
                    stack.enter_context(gpytorch.settings.max_preconditioner_size(0))
                posterior = model(test_x)
                mean = posterior.mean.detach()
                variance = posterior.variance.detach()
                strategy = model.prediction_strategy
                root_rank = strategy.covar_cache.shape[-1]
                if method == "ext":
                    _, q_mat, _ = strategy.cg_lanczos_cache
                    j_resid = None if q_mat is None else q_mat.shape[-1]
                else:
                    j_resid = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        peak_gib = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else float("nan")
        if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
            raise RuntimeError("Non-finite predictive mean or variance")
        cg_warnings = [str(item.message) for item in recorded if issubclass(item.category, NumericalWarning)]
        return {
            "time": elapsed,
            "peak_gib": peak_gib,
            "j_resid": j_resid,
            "root_rank": root_rank,
            "mean": mean.cpu(),
            "variance": variance.cpu(),
            "min_variance": variance.min().item(),
            "cg_warning": " | ".join(cg_warnings),
        }

    output = args.output.expanduser()
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = run_metadata(args)
    completed = set() if args.overwrite else completed_cases(output, metadata)
    write_header = args.overwrite or not output.exists()
    mode = "w" if args.overwrite else "a"

    print("gpytorch:", gpytorch.__file__)
    print("linear_operator:", linear_operator.__file__)
    print("torch/CUDA:", torch.__version__, torch.version.cuda)
    print("device/dtype:", device, dtype)
    print("n/m:", args.n, args.m)
    print("CG/Lanczos tolerance:", args.cg_tolerance, lanczos_tolerance)
    print("exact reference: disabled; only marginal variance is evaluated")
    print("output:", output, flush=True)

    with output.open(mode, newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
            file.flush()
        for case in cases:
            if case.case_id in completed:
                print(f"case {case.case_id}: already in CSV", flush=True)
                continue
            train_y = targets[case.function]
            order = ("love", "ext") if case.case_id % 2 == 0 else ("ext", "love")
            results = {}
            errors = {}
            for method in order:
                print(
                    f"case {case.case_id}/199 config={case.config_id} {case.tier} "
                    f"{case.kernel}/{case.function}: running {method}",
                    flush=True,
                )
                try:
                    results[method] = run_method(case, train_y, method)
                    result = results[method]
                    print(
                        f"  {method}: {result['time']:.3f}s, J={result['root_rank']}, "
                        f"CG warning={bool(result['cg_warning'])}",
                        flush=True,
                    )
                except Exception as exc:
                    errors[method] = f"{type(exc).__name__}: {exc}"
                    print(f"  {method}: FAILED: {errors[method]}", flush=True)

            row = {
                "case_id": case.case_id,
                **case_fields(case),
                **metadata,
                "order": "/".join(order),
                "status": "ok" if not errors else "failed",
                "error_ext": errors.get("ext", ""),
                "error_love": errors.get("love", ""),
            }
            for method, result in results.items():
                row[f"time_{method}"] = result["time"]
                row[f"peak_gib_{method}"] = result["peak_gib"]
                row[f"J_{method}"] = result["root_rank"]
                row[f"min_variance_{method}"] = result["min_variance"]
                row[f"cg_warning_{method}"] = result["cg_warning"]
            if "ext" in results:
                row["J_resid"] = results["ext"]["j_resid"]
            if len(results) == 2:
                ext, love = results["ext"], results["love"]
                row["rel_time"] = (ext["time"] - love["time"]) / love["time"]
                row["rel_mean"] = relative_error(ext["mean"], love["mean"])
                row["max_abs_mean"] = (ext["mean"] - love["mean"]).abs().max().item()
                row["rel_variance"] = relative_error(ext["variance"], love["variance"])
                row["max_abs_variance"] = (ext["variance"] - love["variance"]).abs().max().item()
            writer.writerow(row)
            file.flush()
            print(
                f"case {case.case_id}: {row['status']}, "
                f"rel_time={row.get('rel_time', float('nan')):.3f}, "
                f"rel_mean={row.get('rel_mean', float('nan')):.3e}, "
                f"rel_variance={row.get('rel_variance', float('nan')):.3e}",
                flush=True,
            )
            del results
            gc.collect()


if __name__ == "__main__":
    main()

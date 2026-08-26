import numpy as np
import matplotlib.pyplot as plt

from src.gp import GP
from src.kernels import RBFKernel
from src.means import ZeroMean
from src.linalg import ReorthogonalizationRule


def make_data(
    *,
    n: int,
    m: int,
    domain: tuple[float, float],
    seed: int,
):
    """Generates one-dimensional training and test data for visualization.

    :param int n: Number of training points.
    :param int m: Number of test points.
    :param tuple domain: Interval used for input locations.
    :param int seed: Random seed.
    :return: Training inputs, training targets, test inputs, and true test values.
    """
    rng = np.random.default_rng(seed)

    X_train = rng.uniform(domain[0], domain[1], (n, 1))
    X_test = np.linspace(domain[0], domain[1], m).reshape(-1, 1)

    def f(x):
        return np.sin(x) + np.cos(3.0 * x) + np.exp(-x**2)

    y_train = f(X_train).reshape(-1)
    y_true = f(X_test).reshape(-1)

    return X_train, y_train, X_test, y_true


def confidence_bounds(mean, variance, scale: float = 2.0):
    #Computes symmetric confidence bounds around a mean curve
    variance = np.maximum(variance, 0.0)
    std = np.sqrt(variance)

    return mean - scale * std, mean + scale * std


def plot_single_panel(
    ax,
    *,
    x_plot,
    X_train,
    y_train,
    y_true,
    mean,
    exact_lower,
    exact_upper,
    approx_lower,
    approx_upper,
    title,
    color,
):
    #Plots one posterior confidence comparison panel
    ax.plot(
        x_plot,
        y_true,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label="true function",
    )

    ax.scatter(
        X_train.squeeze(-1),
        y_train,
        color="black",
        s=10,
        alpha=0.25,
        label="training data",
    )

    ax.plot(
        x_plot,
        mean,
        color="black",
        linewidth=1.8,
        label="predictive mean",
    )

    ax.plot(
        x_plot,
        exact_lower,
        color="red",
        linestyle="--",
        linewidth=1.6,
        label="exact boundary",
    )

    ax.plot(
        x_plot,
        exact_upper,
        color="red",
        linestyle="--",
        linewidth=1.6,
    )

    ax.fill_between(
        x_plot,
        approx_lower,
        approx_upper,
        color=color,
        alpha=0.28,
        label="approx. area",
    )

    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_confidence_comparison(
    *,
    n: int = 100,
    m: int = 1000,
    domain: tuple[float, float] = (-3.0, 3.0),
    cg_J: int = 100,
    lanczos_J: int = 100,
    lengthscale: float = 0.3,
    outputscale: float = 1.0,
    noise: float = 1.0,
    jitter: float = 1e-10,
    cg_reorthogonalization_mode: str = "always",
    cg_reorthogonalization_every: int = 1,
    cg_reorthogonalization_start: int = 5,
    cg_rayleigh_tol: float = 1,
    cg_norm_tol: float = 1,
    lanczos_reorthogonalize: bool = True,
    seed: int = 123,
):
    """Plots confidence regions for LOVE, CG-Cholesky, and CG-QR approximations.

    The exact posterior mean and exact confidence boundary are shown in every
    panel. Each panel then overlays one approximate posterior confidence region.

    :param int n: Number of training points. (Default: `100`.)
    :param int m: Number of test points. (Default: `1000`.)
    :param tuple domain: Plotting and sampling interval. (Default: `(-3.0, 3.0)`.)
    :param int cg_J: Maximum number of CG iterations. (Default: `100`.)
    :param int lanczos_J: Maximum number of Lanczos iterations for LOVE. (Default: `100`.)
    :param float lengthscale: RBF kernel lengthscale. (Default: `0.3`.)
    :param float outputscale: RBF kernel outputscale. (Default: `1.0`.)
    :param float noise: Observation noise. (Default: `1.0`.)
    :param float jitter: Jitter used in approximate covariance corrections. (Default: `1e-10`.)
    :param str cg_reorthogonalization_mode: Reorthogonalization mode for CG directions. (Default: `"always"`.)
    :param int cg_reorthogonalization_every: Frequency used when mode is `"every"`. (Default: `1`.)
    :param int cg_reorthogonalization_start: First iteration where reorthogonalization is allowed. (Default: `5`.)
    :param float cg_rayleigh_tol: Rayleigh threshold used when mode is `"rayleigh"`.
    :param float cg_norm_tol: Direction norm threshold used when mode is `"norm"`.
    :param bool lanczos_reorthogonalize: If True, reorthogonalize Lanczos vectors. (Default: True.)
    :param int seed: Random seed. (Default: `123`.)
    """
    X_train, y_train, X_test, y_true = make_data(
        n=n,
        m=m,
        domain=domain,
        seed=seed,
    )

    kernel = RBFKernel(lengthscale=lengthscale, outputscale=outputscale)
    mean = ZeroMean()

    cg_reorthogonalization_rule = ReorthogonalizationRule(
        mode=cg_reorthogonalization_mode,
        every=cg_reorthogonalization_every,
        start=cg_reorthogonalization_start,
        rayleigh_tol=cg_rayleigh_tol,
        norm_tol=cg_norm_tol,
    )

    gp_exact = GP(X_train, y_train, kernel=kernel, mean=mean, noise=noise)
    gp_exact.compute_posterior(method="exact")

    pred_mean = gp_exact.predict_mean(X_test)
    exact_variance = gp_exact.predict_variance(X_test)
    exact_lower, exact_upper = confidence_bounds(pred_mean, exact_variance)

    gp_love = GP(X_train, y_train, kernel=kernel, mean=mean, noise=noise)
    gp_love.compute_posterior(
        method="love",
        cg_J=cg_J,
        lanczos_J=lanczos_J,
        jitter=jitter,
        lanczos_reorthogonalize=lanczos_reorthogonalize,
    )

    love_variance = gp_love.predict_variance(X_test)
    love_lower, love_upper = confidence_bounds(pred_mean, love_variance)

    gp_cg = GP(X_train, y_train, kernel=kernel, mean=mean, noise=noise)
    gp_cg.compute_posterior(
        method="cg",
        cg_J=cg_J,
        jitter=jitter,
        cg_reorthogonalization_rule=cg_reorthogonalization_rule,
    )

    chol_variance = gp_cg.predict_variance(
        X_test,
        cg_correction_method="cholesky",
    )
    chol_lower, chol_upper = confidence_bounds(pred_mean, chol_variance)

    qr_variance = gp_cg.predict_variance(
        X_test,
        cg_correction_method="qr",
    )
    qr_lower, qr_upper = confidence_bounds(pred_mean, qr_variance)

    x_plot = X_test.squeeze(-1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

    plot_single_panel(
        axes[0],
        x_plot=x_plot,
        X_train=X_train,
        y_train=y_train,
        y_true=y_true,
        mean=pred_mean,
        exact_lower=exact_lower,
        exact_upper=exact_upper,
        approx_lower=love_lower,
        approx_upper=love_upper,
        title=f"LOVE, J={gp_love.Q.shape[1]}",
        color="tab:orange",
    )

    plot_single_panel(
        axes[1],
        x_plot=x_plot,
        X_train=X_train,
        y_train=y_train,
        y_true=y_true,
        mean=pred_mean,
        exact_lower=exact_lower,
        exact_upper=exact_upper,
        approx_lower=chol_lower,
        approx_upper=chol_upper,
        title=f"CG-Cholesky, J={gp_cg.D.shape[1]}",
        color="tab:green",
    )

    plot_single_panel(
        axes[2],
        x_plot=x_plot,
        X_train=X_train,
        y_train=y_train,
        y_true=y_true,
        mean=pred_mean,
        exact_lower=exact_lower,
        exact_upper=exact_upper,
        approx_lower=qr_lower,
        approx_upper=qr_upper,
        title=f"CG-QR, J={gp_cg.D.shape[1]}",
        color="tab:blue",
    )

    axes[0].set_ylabel("y")

    for ax in axes:
        ax.set_xlabel("x")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        frameon=False,
    )

    fig.suptitle(
        "Posterior confidence areas "
        f"(lengthscale={lengthscale}, noise={noise}, jitter={jitter})",
        y=1.04,
    )

    fig.tight_layout()
    plt.show()

    print("LOVE effective rank:", gp_love.Q.shape[1])
    print("CG effective rank:", gp_cg.D.shape[1])
    print("Exact variance min/max:", np.min(exact_variance), np.max(exact_variance))
    print("LOVE variance min/max:", np.min(love_variance), np.max(love_variance))
    print("CG-Cholesky variance min/max:", np.min(chol_variance), np.max(chol_variance))
    print("CG-QR variance min/max:", np.min(qr_variance), np.max(qr_variance))


if __name__ == "__main__":
    plot_confidence_comparison()
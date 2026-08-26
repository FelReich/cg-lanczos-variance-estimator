import numpy as np
import matplotlib.pyplot as plt

from src.corrections import exact_correction, cg_qr_correction, love_correction
from src.gp import GP
from src.kernels import RBFKernel
from src.linalg import cg,  cg_store_lanczos_basis, lanczos_tridiagonalization, extend_lanczos_basis, ReorthogonalizationRule
from src.means import ZeroMean


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
    extension_tol: float = 1e-14,
    seed: int = 123,
) -> None:
    """Plots confidence regions for LOVE, CG-QR, and CG-based Lanczos extension.

    The exact posterior mean and exact confidence boundaries are shown in every
    panel. Each panel overlays one approximate posterior confidence region.
    The extended method first recovers a Lanczos basis from CG residuals and then
    continues this basis with additional Lanczos steps.

    :param int n: Number of training points. (Default: `100`.)
    :param int m: Number of test points. (Default: `1000`.)
    :param tuple domain: Sampling and plotting interval. (Default: `(-3.0, 3.0)`.)
    :param int cg_J: Maximum number of CG iterations. (Default: `100`.)
    :param int lanczos_J: Maximum number of Lanczos iterations for LOVE. (Default: `100`.)
    :param float lengthscale: RBF kernel lengthscale. (Default: `0.3`.)
    :param float outputscale: RBF kernel outputscale. (Default: `1.0`.)
    :param float noise: Observation noise. (Default: `1.0`.)
    :param float jitter: Jitter used in approximate covariance corrections. (Default: `1e-10`.)
    :param float extension_tol: Breakdown tolerance for extending the stored Lanczos basis. (Default: `1e-14`.)
    :param int seed: Random seed. (Default: `123`.)
    """
    rng = np.random.default_rng(seed)

    X_train = rng.uniform(domain[0], domain[1], size=(n, 1))
    X_test = np.linspace(domain[0], domain[1], m).reshape(-1, 1)

    def f(x):
        return np.sin(x) + np.cos(3 * x) + np.exp(-x**2)

    y_train = f(X_train).reshape(-1)
    true_y = f(X_test).reshape(-1)

    kernel = RBFKernel(lengthscale=lengthscale, outputscale=outputscale)
    mean = ZeroMean()

    gp = GP(X_train, y_train, kernel=kernel, mean=mean, noise=noise)
    gp.compute_posterior(method="exact")

    K_test = gp.prior_covariance(X_test)
    k = gp.train_test_covariance(X_test)

    exact_corr = exact_correction(gp.K_noise, k)
    exact_cov = K_test - exact_corr
    exact_cov = 0.5 * (exact_cov + exact_cov.T)

    exact_var = np.maximum(np.diag(exact_cov), 0.0)
    exact_std = np.sqrt(exact_var)

    pred_mean = gp.predict_mean(X_test)

    exact_lower = pred_mean - 2.0 * exact_std
    exact_upper = pred_mean + 2.0 * exact_std

    rule = ReorthogonalizationRule(mode="always")

    _, D_qr, KD_qr = cg(
        lambda v: gp.K_noise @ v,
        gp.centered_y,
        J=cg_J,
        tol=1e-6,
        save_directions=True,
        reorthogonalization_rule=rule,
    )

    _, Q_resid, KQ_resid = cg_store_lanczos_basis(
        lambda v: gp.K_noise @ v,
        gp.centered_y,
        J=cg_J,
        tol=1e-6,
        reorthogonalize=True,
    )

    Q_love, T_love = lanczos_tridiagonalization(
        lambda v: gp.K_noise @ v,
        gp.centered_y,
        num_iter=lanczos_J,
        tol=1e-6,
        reorthogonalize=True,
    )

    target_J = Q_resid.shape[1] + Q_love.shape[1]

    Q_ext, T_ext = extend_lanczos_basis(
        lambda v: gp.K_noise @ v,
        Q_resid,
        KQ_resid,
        target_J=target_J,
        tol=extension_tol,
    )

    qr_corr = cg_qr_correction(D_qr, KD_qr, k, jitter=jitter)
    ext_corr = love_correction(Q_ext, T_ext, k, jitter=jitter)
    love_corr = love_correction(Q_love, T_love, k, jitter=jitter)

    methods = [
        ("LOVE", love_corr, Q_love.shape[1]),
        ("CG-QR", qr_corr, D_qr.shape[1]),
        ("CG-Extended", ext_corr, Q_ext.shape[1]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

    for ax, (title, corr, rank) in zip(axes, methods):
        cov = K_test - corr
        cov = 0.5 * (cov + cov.T)

        var = np.maximum(np.diag(cov), 0.0)
        std = np.sqrt(var)

        lower = pred_mean - 2.0 * std
        upper = pred_mean + 2.0 * std

        ax.plot(X_test.squeeze(), true_y, "k--", lw=2, label="true function")
        ax.scatter(X_train.squeeze(), y_train, s=16, color="black", alpha=0.25, label="training data")
        ax.plot(X_test.squeeze(), pred_mean, color="black", lw=2, label="predictive mean")

        ax.plot(X_test.squeeze(), exact_lower, color="red", linestyle="--", lw=2, label="exact boundary")
        ax.plot(X_test.squeeze(), exact_upper, color="red", linestyle="--", lw=2)

        ax.fill_between(
            X_test.squeeze(),
            lower,
            upper,
            color="tab:orange",
            alpha=0.25,
            label="approx. area",
        )

        ax.set_title(f"{title}, J={rank}")
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("y")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5)

    fig.suptitle(
        f"Predictive confidence comparison: lengthscale={lengthscale}, "
        f"noise={noise}, jitter={jitter}",
        y=1.03,
    )

    plt.tight_layout()
    plt.show()

    print("J QR:", D_qr.shape[1])
    print("J residual:", Q_resid.shape[1])
    print("J extended:", Q_ext.shape[1])
    print("J LOVE:", Q_love.shape[1])


if __name__ == "__main__":
    plot_confidence_comparison()
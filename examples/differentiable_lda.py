"""A reproducible regularized Fisher LDA lab using generalized JAX eigh."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import scipy
from scipy import linalg


def make_data(dtype):
    rng = np.random.default_rng(41)
    means = np.array([[-1.8, -0.6, 0.0], [1.8, -0.6, 0.0], [0.0, 1.5, 0.0]])
    covariance = np.array([[1.5, 1.2, 0.3], [1.2, 2.0, 0.5], [0.3, 0.5, 25.0]])
    labels = np.repeat(np.arange(3), 40)
    noise = rng.normal(size=(len(labels), 3)) @ np.linalg.cholesky(covariance).T
    return jnp.asarray(means[labels] + noise, dtype=dtype), jnp.asarray(labels)


def scatter_matrices(x, labels):
    membership = jax.nn.one_hot(labels, 3, dtype=x.dtype)
    counts = membership.sum(axis=0)
    means = membership.T @ x / counts[:, None]
    centered = x - means[labels]
    offsets = means - x.mean(axis=0)
    within = centered.T @ centered / x.shape[0]
    between = (offsets * counts[:, None]).T @ offsets / x.shape[0]
    return within, between


def numpy_scatter(x, labels):
    """Calculate the reference with explicit per-class loops."""
    within = np.zeros((x.shape[1], x.shape[1]))
    between = np.zeros_like(within)
    overall = x.mean(axis=0)
    for label in np.unique(labels):
        group = x[labels == label]
        mean = group.mean(axis=0)
        centered = group - mean
        offset = mean - overall
        within += centered.T @ centered
        between += len(group) * np.outer(offset, offset)
    return within / len(x), between / len(x)


def solve_lda(x, labels, ridge):
    within, between = scatter_matrices(x, labels)
    metric = within + ridge * jnp.eye(x.shape[1], dtype=x.dtype)
    values, vectors = jax.scipy.linalg.eigh(between, metric, type=1)
    return values, vectors, between, metric


def check_dtype(dtype):
    x, labels = make_data(dtype)
    x_ref = np.asarray(x, dtype=np.float64)
    labels_ref = np.asarray(labels)
    within_ref, between_ref = numpy_scatter(x_ref, labels_ref)
    within, between = jax.jit(scatter_matrices)(x, labels)
    low_precision = dtype == np.float32
    tolerance = 4e-5 if low_precision else 2e-11
    np.testing.assert_allclose(within, within_ref, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(between, between_ref, rtol=tolerance, atol=tolerance)
    ridge = jnp.asarray(0.2, dtype=dtype)
    metric_ref = within_ref + float(ridge) * np.eye(3)
    expected = linalg.eigh(between_ref, metric_ref, eigvals_only=True)
    # This fixture has only one zero eigenvalue and separated nonzero values.
    assert np.min(np.diff(expected)) > 0.1
    residuals = []
    normalization_errors = []
    for solver in (solve_lda, jax.jit(solve_lda)):
        values, vectors, _, _ = solver(x, labels, ridge)
        np.testing.assert_allclose(values, expected, rtol=tolerance, atol=tolerance)
        v = np.asarray(vectors, dtype=np.float64)
        lhs, rhs = between_ref @ v, (metric_ref @ v) * np.asarray(values)
        residual = float(
            np.linalg.norm(lhs - rhs) / (np.linalg.norm(lhs) + np.linalg.norm(rhs))
        )
        normalization_error = float(np.linalg.norm(v.T @ metric_ref @ v - np.eye(3)))
        assert residual < tolerance
        assert normalization_error < tolerance
        residuals.append(residual)
        normalization_errors.append(normalization_error)

    ridges = jnp.asarray([0.01, 0.1, 1.0], dtype=dtype)
    batched_values = jax.jit(
        jax.vmap(
            lambda rho: jax.scipy.linalg.eigh(
                between,
                within + rho * jnp.eye(3, dtype=dtype),
                type=1,
                eigvals_only=True,
            )
        )
    )(ridges)
    for rho, values in zip(ridges, batched_values):
        reference = linalg.eigh(
            between_ref, within_ref + float(rho) * np.eye(3), eigvals_only=True
        )
        np.testing.assert_allclose(values, reference, rtol=tolerance, atol=tolerance)

    def score(parameters):
        log_ridge, log_feature_scale = parameters
        scales = jnp.asarray([1.0, 0.0, 0.0], dtype=dtype) * log_feature_scale
        sw, sb = scatter_matrices(x * jnp.exp(scales), labels)
        b = sw + jnp.exp(log_ridge) * jnp.eye(3, dtype=dtype)
        values = jax.scipy.linalg.eigh(sb, b, type=1, eigvals_only=True)
        return jnp.log1p(values[-2:]).sum()

    def scipy_score(parameters):
        log_ridge, log_feature_scale = parameters
        scales = np.exp([log_feature_scale, 0.0, 0.0])
        sw, sb = numpy_scatter(x_ref * scales, labels_ref)
        values = linalg.eigh(sb, sw + np.exp(log_ridge) * np.eye(3), eigvals_only=True)
        return float(np.log1p(values[-2:]).sum())

    parameters = jnp.asarray([-1.5, 0.05], dtype=dtype)
    value, gradient = jax.jit(jax.value_and_grad(score))(parameters)
    parameters_ref = np.asarray(parameters, dtype=np.float64)
    step = 1e-5
    reference_gradient = np.array(
        [
            (
                scipy_score(parameters_ref + step * axis)
                - scipy_score(parameters_ref - step * axis)
            )
            / (2 * step)
            for axis in np.eye(2)
        ]
    )
    gradient_tolerance = 1e-3 if low_precision else 2e-8
    np.testing.assert_allclose(
        value, scipy_score(parameters_ref), rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        gradient, reference_gradient, rtol=gradient_tolerance, atol=gradient_tolerance
    )
    return {
        "dtype": np.dtype(dtype).name,
        "eigenvalues": expected.tolist(),
        "max_defining_residual": max(residuals),
        "max_metric_normalization_error": max(normalization_errors),
        "ridge_sweep": np.asarray(ridges).tolist(),
        "ridge_eigenvalues": np.asarray(batched_values).tolist(),
        "spectral_score": float(value),
        "parameters": np.asarray(parameters).tolist(),
        "gradient": np.asarray(gradient).tolist(),
        "scipy_finite_difference": reference_gradient.tolist(),
        "gradient_max_absolute_error": float(
            np.max(np.abs(np.asarray(gradient) - reference_gradient))
        ),
    }


def make_figure(directory):
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "fisher-lda"
    from matplotlib import pyplot as plt

    x, labels = make_data(np.float64 if jax.config.x64_enabled else np.float32)
    _, vectors, _, _ = solve_lda(x, labels, jnp.asarray(0.2, dtype=x.dtype))
    centered = np.asarray(x) - np.asarray(x).mean(axis=0)
    _, pca_vectors = np.linalg.eigh(centered.T @ centered / len(x))
    pca = centered @ pca_vectors[:, -2:][:, ::-1]
    lda = centered @ np.asarray(vectors)[:, -2:][:, ::-1]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained")
    for ax, points, title in zip(
        axes,
        (pca, lda),
        ("PCA: total variance", "Regularized Fisher LDA: labeled separation"),
    ):
        for label, color in enumerate(("#0072B2", "#D55E00", "#009E73")):
            group = points[np.asarray(labels) == label]
            ax.scatter(
                group[:, 0],
                group[:, 1],
                s=23,
                alpha=0.75,
                c=color,
                label=f"Class {label}",
            )
        ax.set(title=title, xlabel="Component 1", ylabel="Component 2")
        ax.grid(alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(loc="upper right", frameon=False)
    figure.suptitle(
        "120 synthetic samples · 3 features · 3 classes · ridge = 0.2", fontsize=12
    )
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "png"):
        destination = directory / f"fisher-lda.{extension}"
        metadata = {"Date": None} if extension == "svg" else {}
        figure.savefig(destination, dpi=160, metadata=metadata)
        if extension == "svg":
            destination.write_text(
                "\n".join(
                    line.rstrip()
                    for line in destination.read_text(encoding="utf-8").splitlines()
                )
                + "\n",
                encoding="utf-8",
            )
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x64", action="store_true")
    parser.add_argument("--plot-dir", type=Path)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", args.x64)
    cases = [check_dtype(np.float32)]
    if args.x64:
        cases.append(check_dtype(np.float64))
    if args.plot_dir is not None:
        make_figure(args.plot_dir)
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "jax": jax.__version__,
                "jaxlib": jaxlib.__version__,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "devices": [str(device) for device in jax.devices()],
                "x64": args.x64,
                "samples": 120,
                "features": 3,
                "classes": 3,
                "cases": cases,
            },
            indent=2,
        )
    )

"""Check generalized Hermitian eigenproblems and their JAX transformations.

Run against the source revision in case-studies/generalized-eigh.md.
No accelerator or model service is required.
"""

from __future__ import annotations

import argparse
import json
import platform

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import scipy
from scipy import linalg


def make_problem(dtype):
    """Build Hermitian A/H and positive-definite B deterministically."""
    rng = np.random.default_rng(23)
    raw = rng.normal(size=(3, 4, 4))
    if np.issubdtype(dtype, np.complexfloating):
        raw = raw + 1j * rng.normal(size=raw.shape)
    a = (raw[0] + raw[0].conj().T) / 2
    h = (raw[1] + raw[1].conj().T) / 4
    b = raw[2] @ raw[2].conj().T / 4 + np.eye(4)
    return tuple(jnp.asarray(x, dtype=dtype) for x in (a, b, h))


def defining_residual(a, b, w, v, problem_type):
    """Check the defining equation, rather than eigenvector signs/phases."""
    if problem_type == 1:
        lhs, rhs = a @ v, (b @ v) * w
    elif problem_type == 2:
        lhs, rhs = a @ b @ v, v * w
    else:
        lhs, rhs = b @ a @ v, v * w
    return float(
        np.linalg.norm(lhs - rhs) / (np.linalg.norm(lhs) + np.linalg.norm(rhs))
    )


def check_problem(dtype, problem_type):
    a, b, h = make_problem(dtype)
    is_low_precision = np.dtype(dtype).itemsize <= (
        8 if np.issubdtype(dtype, np.complexfloating) else 4
    )
    tolerance = 4e-5 if is_low_precision else 2e-11
    real_dtype = np.float64
    reference_dtype = np.complex128 if np.iscomplexobj(a) else real_dtype
    a_ref, b_ref, h_ref = (np.asarray(x).astype(reference_dtype) for x in (a, b, h))

    def solve(matrix, metric):
        return jax.scipy.linalg.eigh(matrix, metric, type=problem_type)

    w, v = solve(a, b)
    w_jit, v_jit = jax.jit(solve)(a, b)
    expected_w = linalg.eigh(a_ref, b_ref, type=problem_type, eigvals_only=True)
    np.testing.assert_allclose(w, expected_w, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(w_jit, expected_w, rtol=tolerance, atol=tolerance)
    residual = max(
        defining_residual(
            a_ref, b_ref, np.asarray(values), np.asarray(vectors), problem_type
        )
        for values, vectors in ((w, v), (w_jit, v_jit))
    )
    assert residual < tolerance, (dtype, problem_type, residual)

    v_ref = np.asarray(v)
    metric_vectors = (
        np.linalg.solve(b_ref, v_ref) if problem_type == 3 else b_ref @ v_ref
    )
    normalization_error = float(
        np.linalg.norm(v_ref.conj().T @ metric_vectors - np.eye(4))
    )
    assert normalization_error < tolerance * 10

    values_only = jax.jit(
        lambda x, y: jax.scipy.linalg.eigh(x, y, type=problem_type, eigvals_only=True)
    )(a, b)
    np.testing.assert_allclose(values_only, expected_w, rtol=tolerance, atol=tolerance)

    batch_a = jnp.stack([a, a + 0.2 * h])
    batch_w, batch_v = jax.jit(jax.vmap(lambda x: solve(x, b)))(batch_a)
    for matrix, values, vectors in zip(batch_a, batch_w, batch_v):
        expected = linalg.eigh(
            np.asarray(matrix).astype(reference_dtype),
            b_ref,
            type=problem_type,
            eigvals_only=True,
        )
        np.testing.assert_allclose(values, expected, rtol=tolerance, atol=tolerance)
        assert (
            defining_residual(
                np.asarray(matrix),
                np.asarray(b),
                np.asarray(values),
                np.asarray(vectors),
                problem_type,
            )
            < tolerance
        )

    # Differentiate through both A and B. The fixture has separated eigenvalues;
    # it does not assert differentiability at repeated eigenvalues.
    def spectral_objective(parameters):
        theta, phi = parameters
        values = jax.scipy.linalg.eigh(
            a + theta * h,
            b + phi * jnp.eye(4, dtype=b.dtype),
            type=problem_type,
            eigvals_only=True,
        )
        return jnp.log1p(values**2).sum()

    def reference_objective(parameters):
        theta, phi = parameters
        values = linalg.eigh(
            a_ref + theta * h_ref,
            b_ref + phi * np.eye(4),
            type=problem_type,
            eigvals_only=True,
        )
        return float(np.log1p(values**2).sum())

    parameters = np.array([0.1, 0.2], dtype=np.asarray(a).real.dtype)
    gradient = np.asarray(
        jax.jit(jax.grad(spectral_objective))(jnp.asarray(parameters))
    )
    step = 1e-5
    finite_difference = np.array(
        [
            (
                reference_objective(parameters + step * direction)
                - reference_objective(parameters - step * direction)
            )
            / (2 * step)
            for direction in np.eye(2)
        ]
    )
    gradient_tolerance = 5e-4 if is_low_precision else 3e-7
    np.testing.assert_allclose(
        gradient, finite_difference, rtol=gradient_tolerance, atol=gradient_tolerance
    )

    invalid_b = jnp.diag(jnp.asarray([1.0, 1.0, 1.0, -1.0], dtype=dtype))
    invalid_values = jax.scipy.linalg.eigh(
        a, invalid_b, type=problem_type, eigvals_only=True
    )
    assert np.isnan(np.asarray(invalid_values)).all()

    return {
        "dtype": np.dtype(dtype).name,
        "problem_type": problem_type,
        "relative_defining_residual": residual,
        "normalization_error": normalization_error,
        "gradient": gradient.tolist(),
        "scipy_finite_difference": finite_difference.tolist(),
        "gradient_max_absolute_error": float(
            np.max(np.abs(gradient - finite_difference))
        ),
    }


def run_lab(enable_x64=False):
    jax.config.update("jax_enable_x64", enable_x64)
    dtypes = [np.float32, np.complex64]
    if enable_x64:
        dtypes += [np.float64, np.complex128]
    cases = [
        check_problem(dtype, problem_type)
        for dtype in dtypes
        for problem_type in (1, 2, 3)
    ]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "devices": [str(device) for device in jax.devices()],
        "x64": enable_x64,
        "checked_cases": len(cases),
        "cases": cases,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x64", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_lab(args.x64), indent=2))

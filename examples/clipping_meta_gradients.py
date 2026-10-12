"""Reproduce a zero-update reverse-gradient bug and check a masked fix.

This is a vector-only educational reduction of Optax global-norm clipping,
not a replacement optimizer. See articles/clipping-meta-gradients-ru.md.
The clipping expression is adapted from Optax (Apache-2.0); see NOTICE.
"""

from __future__ import annotations

import json
import platform

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np


def baseline_clip(x):
    norm = jnp.sqrt(jnp.vdot(x, x).real)
    return jax.lax.select(norm < 1.0, x, x / norm)


def equality_only_clip(x):
    norm = jnp.sqrt(jnp.vdot(x, x).real)
    return jax.lax.select(norm <= 1.0, x, x / norm)


def guard_after_sqrt_clip(x):
    norm = jnp.sqrt(jnp.vdot(x, x).real)
    trigger = norm < 1.0
    denominator = jnp.where(trigger, 1.0, norm)
    return jax.lax.select(trigger, x, x / denominator)


def fixed_clip(x):
    norm_squared = jnp.vdot(x, x).real
    trigger = jnp.sqrt(norm_squared) < 1.0
    denominator = jnp.sqrt(jnp.where(trigger, 1.0, norm_squared))
    return jax.lax.select(trigger, x, x / denominator)


def outer_loss(initial, clip):
    params = initial
    for _ in range(3):
        grads = jax.grad(lambda p: jnp.sum(p**2) / 2)(params)
        params = params - 0.1 * clip(grads)
    return jnp.sum((params - 1) ** 2)


def numpy_outer_loss(initial):
    """Independent float64 forward calculation, without JAX or autodiff."""
    params = np.array(initial, dtype=np.float64, copy=True)
    for _ in range(3):
        norm = np.linalg.norm(params)
        updates = params if norm < 1.0 else params / norm
        params = params - 0.1 * updates
    return np.sum((params - 1) ** 2)


def run_lab():
    zero = jnp.zeros(3, dtype=jnp.float32)
    identity = np.eye(3, dtype=np.float32)
    expected = np.full(3, -2 * 0.9**3, dtype=np.float64)
    step = 1e-5
    finite_difference = np.array(
        [
            (numpy_outer_loss(step * direction) - numpy_outer_loss(-step * direction))
            / (2 * step)
            for direction in np.eye(3)
        ]
    )
    np.testing.assert_allclose(finite_difference, expected, atol=2e-9, rtol=0)

    observations = []
    for compiled in (False, True):
        transform = jax.jit if compiled else lambda f: f
        for clip in (baseline_clip, equality_only_clip, guard_after_sqrt_clip):
            np.testing.assert_array_equal(transform(clip)(zero), zero)
            reverse = transform(jax.jacrev(clip))(zero)
            assert np.isnan(np.asarray(reverse)).all(), (clip.__name__, reverse)
        np.testing.assert_array_equal(
            transform(jax.jacfwd(baseline_clip))(zero), identity
        )
        baseline_gradient = transform(jax.grad(lambda p: outer_loss(p, baseline_clip)))(
            zero
        )
        assert np.isnan(np.asarray(baseline_gradient)).all()

        for derivative in (jax.jacfwd, jax.jacrev):
            np.testing.assert_array_equal(
                transform(derivative(fixed_clip))(zero), identity
            )
        fixed_gradient = transform(jax.grad(lambda p: outer_loss(p, fixed_clip)))(zero)
        np.testing.assert_allclose(fixed_gradient, expected, atol=1e-7, rtol=1e-6)

        for x in (jnp.array([0.1, 0.2]), jnp.array([3.0, 4.0])):
            reference_x = np.asarray(x, dtype=np.float64)
            norm = np.linalg.norm(reference_x)
            expected_jacobian = (
                np.eye(2)
                if norm < 1.0
                else np.eye(2) / norm - np.outer(reference_x, reference_x) / norm**3
            )
            np.testing.assert_allclose(
                transform(jax.jacrev(fixed_clip))(x),
                expected_jacobian,
                atol=1e-7,
                rtol=1e-6,
            )
            np.testing.assert_array_equal(
                transform(fixed_clip)(x), transform(baseline_clip)(x)
            )
        observations.append(
            {
                "compiled": compiled,
                "baseline_reverse_jacobian_all_nan": True,
                "equality_only_reverse_jacobian_all_nan": True,
                "guard_after_sqrt_reverse_jacobian_all_nan": True,
                "baseline_meta_gradient_all_nan": True,
                "fixed_reverse_jacobian": "identity",
                "fixed_meta_gradient": np.asarray(fixed_gradient).tolist(),
            }
        )

    batched = jax.jit(jax.vmap(jax.jacrev(fixed_clip)))(jnp.zeros((2, 3)))
    np.testing.assert_array_equal(batched, np.broadcast_to(identity, (2, 3, 3)))
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "numpy": np.__version__,
        "devices": [str(d) for d in jax.devices()],
        "dtype": "float32",
        "threshold": 1.0,
        "inner_steps": 3,
        "expected_meta_gradient": expected.tolist(),
        "numpy_float64_finite_difference": finite_difference.tolist(),
        "vmap_reverse_jacobians": "identity",
        "observations": observations,
    }


if __name__ == "__main__":
    print(json.dumps(run_lab(), indent=2, allow_nan=False))

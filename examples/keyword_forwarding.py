"""Measure keyword forwarding in reusable JAX autodiff on a pinned source tree."""

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect", choices=("baseline", "fixed"), required=True)
    args = parser.parse_args()
    jax.config.update("jax_platforms", "cpu")

    def loss(x, scale=1.0):
        return jnp.sum(x * scale)

    x = jnp.ones(3, dtype=jnp.float32)
    forward, backward = jax.fwd_and_bwd(loss, argnums=0)
    value, residuals = forward(x, scale=0.5)
    gradient = backward(residuals, jnp.asarray(1.0, dtype=x.dtype))
    observed = {"value": float(value), "gradient": np.asarray(gradient).tolist()}
    reference = {"value": 1.5, "gradient": [0.5, 0.5, 0.5]}
    baseline = {"value": 3.0, "gradient": [1.0, 1.0, 1.0]}
    expected = baseline if args.expect == "baseline" else reference
    print(json.dumps({"jax_version": jax.__version__, "observed": observed,
                      "reference": reference, "expected_stage": args.expect}, indent=2))
    if observed != expected:
        raise SystemExit("The result does not match the requested stage.")


if __name__ == "__main__":
    main()

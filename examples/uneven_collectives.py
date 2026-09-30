"""Measure uneven-group means, constant sums, and gradients on four CPU devices."""

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
    jax.config.update("jax_num_cpu_devices", 4)

    groups = [[0], [1, 2, 3]]
    values = np.arange(4, dtype=np.float32)
    mean = jax.pmap(
        lambda x: jax.lax.pmean(x, "i", axis_index_groups=groups), axis_name="i"
    )
    constant_sum = jax.pmap(
        lambda _: jax.lax.psum(1, "i", axis_index_groups=groups), axis_name="i"
    )
    observed = {
        "mean": np.asarray(mean(values)).tolist(),
        "constant_sum": np.asarray(constant_sum(values)).tolist(),
        "gradient": np.asarray(jax.grad(lambda x: jnp.sum(mean(x)))(values)).tolist(),
    }
    # Independent reference: assign each NumPy group reduction to its members.
    group_means = np.empty_like(values)
    group_sizes = np.empty(4, dtype=np.int32)
    for group in groups:
        group_means[group] = np.mean(values[group])
        group_sizes[group] = len(group)
    reference = {"mean": group_means.tolist(), "constant_sum": group_sizes.tolist(),
                 "gradient": [1.0] * 4}
    baseline = {"mean": [0.0, 6.0, 6.0, 6.0], "constant_sum": [1, 1, 1, 1],
                "gradient": [1.0, 3.0, 3.0, 3.0]}
    expected = baseline if args.expect == "baseline" else reference
    print(json.dumps({"jax_version": jax.__version__, "device_count": jax.device_count(),
                      "groups": groups, "observed": observed, "reference": reference,
                      "expected_stage": args.expect}, indent=2))
    if observed != expected:
        raise SystemExit("The result does not match the requested stage.")


if __name__ == "__main__":
    main()

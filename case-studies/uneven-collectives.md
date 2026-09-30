# When a constant is not constant across replica groups

By Jabrail Khalil. [Issue #41132](https://github.com/jax-ml/jax/issues/41132) · [Submitted PR #41157](https://github.com/jax-ml/jax/pull/41157).

The contribution is open for review. I reproduced the problem on CPU; GPU and TPU validation remain for upstream CI.

## A four-device example

Assign values `[0, 1, 2, 3]` to four logical CPU devices. Put device 0 in one group and devices 1–3 in another:

```python
groups = [[0], [1, 2, 3]]
f = jax.pmap(
    lambda x: jax.lax.pmean(x, "i", axis_index_groups=groups),
    axis_name="i",
)
```

The first group's mean is 0. The other group's mean is `(1 + 2 + 3) / 3 = 2`. Every member receives its group's reduction, so the answer should be **[0, 2, 2, 2]**. My upstream-baseline reproduction returned **[0, 6, 6, 6]**.

I then checked `psum(1)` using the same groups. It should return each device's group size: **[1, 3, 3, 3]**. It returned **[1, 1, 1, 1]**. This isolates the problem from the input data and from floating-point rounding.

## Follow the shortcut

`pmean` computes a sum and divides it by the group size. The size is obtained through `psum(1)`.

The `psum` constant shortcut multiplied a constant by the length of the first group. That is valid when all groups have the same size. With uneven groups, the value supplied by each device is constant, but the reduction result varies by group. The shortcut erased that distinction.

My fix preserves the shortcut for equal-size groups and lets uneven groups use the existing collective implementation. The implementation can then calculate a different sum for each group.

This exposed a second requirement: the group-size collective produces an integer array. Dividing a float or complex output by that array needs an explicit dtype conversion under strict promotion. I convert the divisor to each output leaf's dtype before dividing.

## Check gradients, not only forward values

The sum of all replicated group means has derivative 1 with respect to every input. Each group member contributes `1 / group_size` to its mean, and that mean is repeated `group_size` times in the sum.

The baseline gradient was **[1, 3, 3, 3]**; the fixed result is **[1, 1, 1, 1]**. A wrong denominator therefore affects both the displayed value and downstream optimization.

The 16 regression scenarios cover reversed group order, scalar/array pytrees, integer/float/bool constants, float/complex means, standard/strict dtype promotion, and gradients. On the base, 14 fail and 2 pass; with the fix, all 16 pass. The two passing baseline cases guard existing bool-constant behavior.

The complete `pmap_test.py` file passed on eight logical CPU devices in both x64 modes: **289 passed, 31 skipped** per mode. All pre-commit hooks passed.

## Scope and upstream history

The documented API supports uneven reduction groups outside TPU; TPU requires equal-size groups. The new tests follow the existing TPU skip convention.

A broader earlier change, [#37941](https://github.com/jax-ml/jax/pull/37941), included similar handling and was reverted by [#37958](https://github.com/jax-ml/jax/pull/37958). I included that history in the PR for reviewers. The public record I inspected does not explain the revert, so I do not infer its cause. This contribution focuses on the reported sum/mean bug and its regression coverage.

Run [the example](../examples/uneven_collectives.py) using the [pinned before/after instructions](../README.md#run-the-beforeafter-examples). The [validation record](../validation/results.json) contains the source checkpoints and logs.

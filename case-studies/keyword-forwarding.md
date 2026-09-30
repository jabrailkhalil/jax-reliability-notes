# A loss coefficient lost between the caller and autodiff

By Jabrail Khalil. [Issue #41121](https://github.com/jax-ml/jax/issues/41121) · [Submitted PR #41156](https://github.com/jax-ml/jax/pull/41156).

The contribution is open for review. This article describes my CPU reproduction and submitted fix, not a released upstream change.

## Start with an answer that is easy to calculate

Consider a loss `sum(x * scale)`, with three input values equal to one and `scale=0.5`. The loss must be **1.5**, and its derivative with respect to each input must be **0.5**.

`jax.fwd_and_bwd` creates reusable forward and backward functions. The forward function returns both the value and the residuals that the backward function needs:

```python
def loss(x, scale=1.0):
    return jnp.sum(x * scale)

forward, backward = jax.fwd_and_bwd(loss, argnums=0)
value, residuals = forward(jnp.ones(3), scale=0.5)
gradient = backward(residuals, jnp.asarray(1.0))
```

On the pinned upstream baseline, I measured a loss of **3.0** and a gradient of **[1, 1, 1]**. Both match the default coefficient, not the coefficient I supplied. A training job using a keyword coefficient could therefore optimize a different loss from the one its caller intended.

## Trace the function call before changing differentiation

The wrapper accepted keyword arguments but supplied an empty dictionary to `argnums_partial2`. This helper selects the positional arguments to differentiate and builds the function that captures the remaining arguments. Passing `{}` meant that the function construction lost the keyword arguments before the forward/backward transformation began.

The production change is one argument:

```diff
- argnums_partial2(fun, argnums, args, {})
+ argnums_partial2(fun, argnums, args, kwargs)
```

The fix preserves the existing transformation and forwards the caller's function arguments. I did not add a separate keyword differentiation mechanism.

## Test the wrapper's contract

I added 12 regression scenarios covering integer and tuple `argnums`, auxiliary outputs, jitted and eager calls, required keyword-only arguments, and rejection of unexpected keywords. The jitted scenarios change the supplied coefficient between calls and verify reuse without another trace. This checks that a keyword's value remains part of the computation rather than becoming an accidental frozen constant.

All 12 new scenarios failed on the unmodified base and passed after the change. The complete `api_test.py` file passed with x64 disabled (**980 passed, 36 skipped**) and enabled (**983 passed, 37 skipped**). All pre-commit hooks passed. These results are Linux CPU checks.

## Takeaway

A numerical API can return perfectly plausible values while computing the wrong function. Before investigating the differentiation rule, verify that the wrapper preserved the caller's positional arguments, keyword arguments, and auxiliary-output contract. An independently calculated value and derivative make a small regression test much stronger.

Run [the example](../examples/keyword_forwarding.py) using the [pinned before/after instructions](../README.md#run-the-beforeafter-examples). The [validation record](../validation/results.json) identifies the submitted source and test evidence.

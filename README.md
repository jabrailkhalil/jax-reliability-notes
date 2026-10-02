# JAX reliability notes

Reproducible investigations of JAX correctness bugs, by Jabrail Khalil.

I follow a small loop: calculate an independent expected result, reproduce the bug on upstream code, understand the transformation involved, and add a regression test before changing the implementation.

## Case studies

| Case | User-visible problem | Upstream contribution |
| --- | --- | --- |
| [Keyword forwarding](case-studies/keyword-forwarding.md) | A supplied loss coefficient is ignored; both the value and gradient are wrong. | [JAX #41156](https://github.com/jax-ml/jax/pull/41156) |
| [Uneven collective groups](case-studies/uneven-collectives.md) | A group mean uses another group's size; constant sums and gradients are also wrong. | [JAX #41157](https://github.com/jax-ml/jax/pull/41157) |
| [Generalized Hermitian eigenproblems](case-studies/generalized-eigh.md) | A missing numerical API needs correct eigenvector transforms, normalization, and derivatives. | [JAX #41162](https://github.com/jax-ml/jax/pull/41162), [#41197](https://github.com/jax-ml/jax/pull/41197) |

As of 2 October 2026, keyword forwarding and both generalized-eigh contributions are **merged**; the uneven-groups fix is **open for upstream review**. The patch copies below demonstrate the original before/after investigations. Their checks used Linux CPU. The new generalized-eigh lab uses a later source revision and was checked on Windows CPU. No accelerator execution is claimed for these labs.

## Run the before/after examples

Use Python 3.12 and Git. These commands pin the upstream source so that a future upstream fix does not silently change the baseline.

```sh
git clone https://github.com/jabrailkhalil/jax-reliability-notes.git
cd jax-reliability-notes
python -m venv .venv
```

Activate the environment:

```sh
# Linux / macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Then, from the repository root:

```sh
git clone --filter=blob:none https://github.com/jax-ml/jax.git .lab/jax
git -C .lab/jax checkout 004eda6fcb36dd11d61ea9fa5fa2249a1810d604
python -m pip install -r requirements-demo.txt -e .lab/jax

python examples/keyword_forwarding.py --expect baseline
python examples/uneven_collectives.py --expect baseline

git -C .lab/jax apply ../../patches/keyword-forwarding.patch
git -C .lab/jax apply ../../patches/uneven-collectives.patch

python examples/keyword_forwarding.py --expect fixed
python examples/uneven_collectives.py --expect fixed
```

The examples print actual results alongside independently calculated references. `--expect baseline` succeeds only when the pinned bug is reproduced. `--expect fixed` succeeds only when all measured results match the references. An unexpected result exits with an error. The collective example configures four logical CPU devices before initializing JAX's backend; it does not require four physical CPUs or an accelerator.

The exact dependency environment used for validation was Python 3.12.14, jaxlib 0.11.2, NumPy 2.5.3, SciPy 1.18.1, and ml_dtypes 0.6.0. The source checkout above supplies JAX itself.

## Test evidence

The [validation directory](validation/) contains the measured example output and complete relevant test-file logs. These are CPU checks; they do not replace upstream accelerator CI.

| Contribution | Before the fix | After the fix | Complete relevant test file, x64 off / on |
| --- | --- | --- | --- |
| Keyword forwarding | 12 new regression scenarios fail | 12 new scenarios pass | `api_test.py`: 980 passed, 36 skipped / 983 passed, 37 skipped |
| Uneven groups | 14 of 16 regression scenarios fail; 2 pass | All 16 pass | `pmap_test.py`: 289 passed, 31 skipped in each mode |

Both submitted contributions passed JAX's complete `pre-commit run --all-files` checks, including Ruff and Pyrefly. The source checkpoints and commands are recorded in [validation/results.json](validation/results.json).

## A short technical session

The [10-minute session outline](talk-outline.md) turns these investigations into a practical walkthrough of testing numerical APIs. It is prepared material; no meetup delivery or audience impact is claimed here.

## Generalized eigenproblems in ML

In [my Russian-language Fisher LDA article](articles/generalized-eigh-lda-ru.md), I use the generalized `eigh` API to build a differentiable dimensionality-reduction example. I compare it with independent SciPy calculations, check `jit` and `vmap`, and verify gradients with finite differences. I include runnable code, measured CPU results, and a plot comparing PCA with regularized LDA on synthetic data. I have prepared the material for a lesson or workshop; I have not delivered it to an audience.

## License

The educational material and examples are licensed under Apache-2.0. The patch excerpts modify JAX, which is also licensed under Apache-2.0; their upstream context remains copyright The JAX Authors. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
